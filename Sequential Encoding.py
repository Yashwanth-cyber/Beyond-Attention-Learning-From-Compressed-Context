import sys
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
import numpy as np

# --- Part 1: Helper Functions & Data Loading ---
def load_word_embeddings(file_path):
    if not os.path.exists(file_path):
        print(f"FATAL ERROR: Embedding file not found at '{file_path}'")
        sys.exit(1)
    print(f"Loading word embeddings from '{file_path}'...")
    word_to_emb = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        try: _, embedding_dim = map(int, f.readline().split())
        except (ValueError, IndexError): sys.exit("FATAL ERROR: Invalid header in embedding file.")
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1: continue
            try: word_to_emb[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError: continue
    print(f"✅ Loaded {len(word_to_emb)} words with embedding size {embedding_dim}.")
    return word_to_emb, embedding_dim

# --- Part 2: The Stable Symmetrical Autoencoder Architecture ---
def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)

class NormResBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.block = nn.Sequential(nn.Linear(features, features), nn.GELU(), nn.Linear(features, features))
    def forward(self, x): return self.block(self.norm(x)) + x

class SymmetricalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SymmetricalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 192), NormResBlock(192),
            nn.Linear(192, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 192), NormResBlock(192),
            nn.Linear(192, input_dim))
        self.apply(kaiming_init_weights)
    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x): return self.encoder(x)
    # --- THE FIX IS HERE: The model needs a decode method to be called ---
    def decode(self, x): return self.decoder(x)

# --- Part 3: The Main Training Pipeline ---
def main_training_pipeline(embedding_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Running on device: {device}")

    word_to_emb, embedding_dim = load_word_embeddings(embedding_file_path)
    if embedding_dim != 128: sys.exit(f"FATAL ERROR: Word embedding size must be 128.")

    special_tokens = ['<SOS>', '<EOS>']
    for token in special_tokens:
        if token not in word_to_emb:
            word_to_emb[token] = np.random.uniform(-0.1, 0.1, embedding_dim).astype(np.float32)

    print("\nLoading and processing dataset...")
    df = pd.read_parquet("hf://datasets/Dahoas/sft-gptj-synthetic-prompt-responses/data/train-00000-of-00001-5588d64a331985b4.parquet")

    model = SymmetricalAutoencoder(input_dim=2 * embedding_dim, latent_dim=embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True)
    model_checkpoint_dir = "recurrent_ae_checkpoints"
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # ========================== PHASE 1: PRE-TRAINING (BIGRAMS) ==========================
    print("\n--- Starting Phase 1: Pre-training on 2-Word Chunks ---")

    full_text = " ".join(df['prompt'].tolist() + df['response'].tolist())
    full_sequence = [torch.from_numpy(word_to_emb[word]) for word in re.findall(r'\b\w+\b', full_text.lower()) if word in word_to_emb]

    class BigramDataset(Dataset):
        def __init__(self, sequence): self.sequence = sequence
        def __len__(self): return len(self.sequence) - 1
        def __getitem__(self, idx): return torch.cat([self.sequence[idx], self.sequence[idx+1]])

    bigram_dataset = BigramDataset(full_sequence)
    bigram_loader = DataLoader(bigram_dataset, batch_size=256, shuffle=True, pin_memory=True if device.type=='cuda' else False)

    for epoch in range(1, 3):
        pbar = tqdm(bigram_loader, desc=f"Pre-training Epoch {epoch}/2")
        total_epoch_loss, total_epoch_accuracy = 0, 0
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch) # Calling model.forward()
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            with torch.no_grad():
                cosine_sim = F.cosine_similarity(reconstructed, batch, dim=1).mean().item()
                accuracy = (cosine_sim + 1.0) / 2.0 * 100.0
                total_epoch_accuracy += accuracy

            pbar.set_postfix({'Loss': f"{total_epoch_loss / (pbar.n + 1):.6f}", 'Acc': f"{total_epoch_accuracy / (pbar.n + 1):.2f}%"})

        avg_loss = total_epoch_loss / len(pbar)
        scheduler.step(avg_loss)
        checkpoint_path = os.path.join(model_checkpoint_dir, f'model_pretrain_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch}/2 Complete — Avg Loss: {avg_loss:.6f} — Avg Acc: {total_epoch_accuracy / len(pbar):.2f}%")
        print(f"✅ Pre-training checkpoint saved to '{checkpoint_path}'")

    print("✅ Pre-training complete.")

    # ========================== PHASE 2: FINE-TUNING (RECURRENT) ==========================
    print("\n--- Starting Phase 2: Fine-tuning with Recurrent Logic ---")

    class SequenceDataset(Dataset):
        def __init__(self, df, word_to_emb):
            texts = df['prompt'].tolist() + df['response'].tolist()
            self.sequences = []
            for text in tqdm(texts, desc="Processing fine-tune data"):
                seq_words = ['<SOS>'] + re.findall(r'\b\w+\b', text.lower()) + ['<EOS>']
                seq = [torch.from_numpy(word_to_emb[w]) for w in seq_words if w in word_to_emb]
                if len(seq) > 2: self.sequences.append(torch.stack(seq))
        def __len__(self): return len(self.sequences)
        def __getitem__(self, idx): return self.sequences[idx]

    sequence_dataset = SequenceDataset(df, word_to_emb)
    def collate_fn(batch): return pad_sequence(batch, batch_first=True, padding_value=0.0)
    finetune_loader = DataLoader(sequence_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    for epoch in range(1, 3):
        pbar = tqdm(finetune_loader, desc=f"Fine-tuning Epoch {epoch}/2")
        total_epoch_loss, total_epoch_accuracy, steps = 0, 0, 0

        for batch in pbar:
            batch = batch.to(device)
            batch_size, seq_len, _ = batch.shape

            # Initialize latent_vectors for the batch
            initial_pairs = torch.cat([batch[:, 0, :], batch[:, 1, :]], dim=1)
            latent_vectors = model.encode(initial_pairs)

            # Loop through the rest of the sequence, fully batched
            for t in range(2, seq_len):
                current_word_embeddings = batch[:, t, :]
                mask = batch[:, t, :].abs().sum(dim=1) != 0 # Find non-padded steps
                if mask.sum() == 0: continue

                recurrent_input = torch.cat([latent_vectors[mask].detach(), current_word_embeddings[mask]], dim=1)

                optimizer.zero_grad()
                new_latent_vectors = model.encode(recurrent_input)
                # --- THE FIX IS HERE: Call decode on the new latent vector ---
                reconstructed = model.decode(new_latent_vectors)
                loss = criterion(reconstructed, recurrent_input)
                loss.backward()
                optimizer.step()

                # Update latent vectors for the next step only for active sequences
                latent_vectors[mask] = new_latent_vectors

                total_epoch_loss += loss.item()
                with torch.no_grad():
                    cosine_sim = F.cosine_similarity(reconstructed, recurrent_input, dim=1).mean().item()
                    total_epoch_accuracy += (cosine_sim + 1.0) / 2.0 * 100.0
                steps += 1

            if steps > 0:
                pbar.set_postfix({
                    'Avg Loss': f"{total_epoch_loss / steps:.6f}",
                    'Avg Acc': f"{total_epoch_accuracy / steps:.2f}%"})

        avg_loss = total_epoch_loss / steps if steps > 0 else 0
        scheduler.step(avg_loss)
        checkpoint_path = os.path.join(model_checkpoint_dir, f'model_finetune_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch}/2 Complete — Avg Step Loss: {avg_loss:.6f} — Avg Step Acc: {total_epoch_accuracy / steps if steps > 0 else 0:.2f}%")
        print(f"✅ Fine-tuning checkpoint saved to '{checkpoint_path}'")

    print("\n✅✅✅ Hierarchical training complete! ✅✅✅")

if __name__ == '__main__':
    embedding_file_path = '/content/Word_Embeddings (1).vec'

    if not os.path.exists(embedding_file_path):
        dummy_emb_path = 'dummy_embeddings_128d.vec'
        print(f"'{embedding_file_path}' not found.")
        if not os.path.exists(dummy_emb_path):
            print("Creating dummy embeddings...")
            dummy_vocab = ['<SOS>', '<EOS>'] + [f'word_{i}' for i in range(20000)]
            with open(dummy_emb_path, 'w') as f:
                f.write(f"{len(dummy_vocab)} 128\n")
                for word in dummy_vocab: f.write(f"{word} {' '.join(map(str, np.random.uniform(-1,1,128).astype(np.float32)))}\n")
        embedding_file_path = dummy_emb_path

    main_training_pipeline(embedding_file_path)
