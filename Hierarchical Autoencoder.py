import sys
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import requests
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
        try:
            _, embedding_dim = map(int, f.readline().split())
        except (ValueError, IndexError):
            print("FATAL ERROR: Invalid header in embedding file.")
            sys.exit(1)
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1: continue
            word = parts[0]
            try:
                word_to_emb[word] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError: continue
    print(f"✅ Loaded {len(word_to_emb)} words with embedding size {embedding_dim}.")
    return word_to_emb, embedding_dim

def text_to_sequence(text, word_to_emb):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word_to_emb[word] for word in words if word in word_to_emb]

# --- Part 2: The Stable Hierarchical Autoencoder Architecture ---
def kaiming_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)

class NormResBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.block = nn.Sequential(nn.Linear(features, features), nn.GELU(), nn.Linear(features, features))
    def forward(self, x):
        return self.block(self.norm(x)) + x

class HierarchicalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(HierarchicalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), NormResBlock(512),
            nn.Linear(512, 256), NormResBlock(256),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), NormResBlock(256),
            nn.Linear(256, 512), NormResBlock(512),
            nn.Linear(512, input_dim)
        )
        self.apply(kaiming_init_weights)
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def encode(self, x):
        return self.encoder(x)

# --- Part 3: The Main Training Pipeline ---
def main_training_pipeline(embedding_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Running on device: {device}")

    word_to_emb, embedding_dim = load_word_embeddings(embedding_file_path)
    if embedding_dim != 128:
        print(f"FATAL ERROR: Word embedding size is {embedding_dim}, but the architecture requires 128.")
        sys.exit(1)

    print("\nLoading and processing dataset...")
    df = pd.read_parquet("hf://datasets/Dahoas/sft-gptj-synthetic-prompt-responses/data/train-00000-of-00001-5588d64a331985b4.parquet")
    full_text = " ".join(df['prompt'].tolist() + df['response'].tolist())
    full_sequence = text_to_sequence(full_text, word_to_emb)
    print(f"Dataset converted to a single sequence of {len(full_sequence)} word embeddings.")

    latent_dim = 128
    model = HierarchicalAutoencoder(input_dim=4 * embedding_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True)
    model_checkpoint_dir = "stable_hierarchical_checkpoints"
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    def train_stage(stage_name, chunk_size, epochs):
        print(f"\n--- Starting Training Stage: {stage_name} (Chunk Size: {chunk_size}) ---")

        num_chunks = len(full_sequence) // chunk_size
        chunks = [torch.tensor(full_sequence[i*chunk_size : (i+1)*chunk_size], dtype=torch.float32) for i in range(num_chunks)]
        chunk_dataset = TensorDataset(torch.stack(chunks))
        chunk_loader = DataLoader(chunk_dataset, batch_size=128, shuffle=True)

        for epoch in range(1, epochs + 1):
            model.train()
            # --- THE CHANGE IS HERE: tqdm provides the visual progress bar ---
            pbar = tqdm(chunk_loader, desc=f"Epoch {epoch}/{epochs} [{stage_name}]", leave=False)
            total_loss = 0
            total_accuracy = 0

            for (batch,) in pbar:
                batch = batch.to(device)

                def process_batch(sub_batch):
                    if sub_batch.shape[1] == 4:
                        return sub_batch.view(sub_batch.shape[0], -1)
                    sub_chunk_size = sub_batch.shape[1] // 4
                    parts = []
                    for i in range(4):
                        part = sub_batch[:, i*sub_chunk_size : (i+1)*sub_chunk_size, :]
                        processed_part = process_batch(part)
                        with torch.no_grad():
                            encoded_part = model.encode(processed_part)
                        parts.append(encoded_part)
                    return torch.cat(parts, dim=1)

                input_vectors = process_batch(batch)

                optimizer.zero_grad()
                reconstructed_vectors = model(input_vectors)
                loss = criterion(reconstructed_vectors, input_vectors)
                loss.backward()
                optimizer.step()

                # --- Update running metrics ---
                total_loss += loss.item()
                with torch.no_grad():
                    cosine_sim = F.cosine_similarity(reconstructed_vectors, input_vectors, dim=1).mean().item()
                    scaled_accuracy = (cosine_sim + 1.0) / 2.0 * 100.0
                    total_accuracy += scaled_accuracy

                # --- THE CHANGE IS HERE: This updates the metrics on the progress bar in real-time ---
                pbar.set_postfix({
                    'Loss': f"{total_loss / (pbar.n + 1):.4f}",
                    'Accuracy': f"{total_accuracy / (pbar.n + 1):.2f}%"
                })

            # Close the progress bar for this epoch
            pbar.close()

            avg_loss = total_loss / len(chunk_loader)
            avg_accuracy = total_accuracy / len(chunk_loader)
            scheduler.step(avg_loss)

            # Print a clean summary line for the completed epoch
            print(f"Epoch {epoch}/{epochs} Complete — Avg Loss: {avg_loss:.6f} — Avg Accuracy: {avg_accuracy:.2f}%")

            checkpoint_path = os.path.join(model_checkpoint_dir, f'model_{stage_name}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Model checkpoint saved to '{checkpoint_path}'")

    # --- Execute the Hierarchical Training Curriculum ---
    train_stage(stage_name="4-Word-Chunks", chunk_size=4, epochs=3)
    train_stage(stage_name="16-Word-Chunks", chunk_size=16, epochs=2)
    train_stage(stage_name="64-Word-Chunks", chunk_size=64, epochs=2)

    print("\n✅✅✅ Hierarchical training complete! ✅✅✅")

if __name__ == '__main__':
    embedding_file_path = '/content/gptj_word_embeddings.vec'

    if not os.path.exists(embedding_file_path):
        print(f"'{embedding_file_path}' not found. Please provide a valid path.")
        if not os.path.exists('dummy_embeddings_128d.vec'):
            print("Creating a dummy 128d embedding file: 'dummy_embeddings_128d.vec'")
            DUMMY_VOCAB_SIZE, DUMMY_EMBED_DIM = 20000, 128
            with open('dummy_embeddings_128d.vec', 'w', encoding='utf-8') as f:
                f.write(f"{DUMMY_VOCAB_SIZE} {DUMMY_EMBED_DIM}\n")
                for i in range(DUMMY_VOCAB_SIZE):
                    word = f"word_{i}"
                    vector = ' '.join(map(str, np.random.uniform(-1, 1, DUMMY_EMBED_DIM).astype(np.float32)))
                    f.write(f"{word} {vector}\n")
            embedding_file_path = 'dummy_embeddings_128d.vec'

    main_training_pipeline(embedding_file_path)
