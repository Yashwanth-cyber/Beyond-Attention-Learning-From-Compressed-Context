import sys
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import pandas as pd
import numpy as np

# --- Part 1: All Prerequisite Architectures & Helper Functions ---
# (Architectures remain the same)
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

class HierarchicalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(HierarchicalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), NormResBlock(512),
            nn.Linear(512, 256), NormResBlock(256),
            nn.Linear(256, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), NormResBlock(256),
            nn.Linear(256, 512), NormResBlock(512),
            nn.Linear(512, input_dim))
        self.apply(kaiming_init_weights)
    def encode(self, x): return self.encoder(x)

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
    def encode(self, x): return self.encoder(x)

class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(ClassifierHead, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, vocab_size))
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight);
            if module.bias is not None: nn.init.constant_(module.bias, 0)
    def forward(self, x): return self.network(x)

def encode_prompt_to_context_vector(text_sequence, word_to_emb, context_encoder, device):
    with torch.no_grad():
        words = re.findall(r'\b\w+\b', text_sequence.lower())
        sequence_tensors = [torch.from_numpy(word_to_emb[word]).to(device) for word in words if word in word_to_emb]
        if not sequence_tensors: return torch.zeros(128, device=device)
        current_level_tensors = sequence_tensors
        while len(current_level_tensors) > 1:
            num_to_pad = (4 - (len(current_level_tensors) % 4)) % 4
            if num_to_pad > 0: current_level_tensors.extend([torch.zeros_like(current_level_tensors[0])] * num_to_pad)
            chunks = [current_level_tensors[i:i+4] for i in range(0, len(current_level_tensors), 4)]
            L_input = torch.stack([torch.cat(chunk, dim=0) for chunk in chunks])
            L_output = context_encoder.encode(L_input)
            current_level_tensors = list(L_output)
        return current_level_tensors[0]

# --- Part 2: DataProcessor (Unchanged) ---
class DataProcessor:
    def __init__(self, embedding_file_path, device, embedding_dim=128):
        self.device = device
        self.word_to_emb, self.embedding_dim = self._load_word_embeddings(embedding_file_path, embedding_dim)
        special_tokens = ['< SOS >', '<EOS>', '<PAD>']
        for token in special_tokens:
            if token not in self.word_to_emb:
                self.word_to_emb[token] = np.random.uniform(-0.1, 0.1, self.embedding_dim).astype(np.float32)
        self.vocab = list(self.word_to_emb.keys())
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        embedding_weights = torch.tensor(np.stack([self.word_to_emb[word] for word in self.vocab], axis=0), dtype=torch.float)
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=self.word_to_idx['<PAD>']).to(self.device)

    def _load_word_embeddings(self, file_path, target_embedding_dim):
        # ... (loading logic is fine, no changes needed)
        if not os.path.exists(file_path): sys.exit(f"FATAL ERROR: Embedding file not found at '{file_path}'")
        print(f"Loading word embeddings from '{file_path}'...")
        word_to_emb = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            try: _, embedding_dim = map(int, f.readline().split())
            except (ValueError, IndexError): sys.exit("FATAL ERROR: Invalid header.")
            if embedding_dim != target_embedding_dim: sys.exit(f"FATAL: Word embedding size must be {target_embedding_dim}.")
            for line in f:
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1: continue
                try: word_to_emb[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                except ValueError: continue
        print(f"✅ Loaded {len(word_to_emb)} words.")
        return word_to_emb, embedding_dim

    def get_pretrain_loader(self, df, batch_size):
        # ... (loader logic is fine, no changes needed)
        class PretrainDataset(Dataset):
            def __init__(self, df, word_to_idx):
                texts = df['prompt'].tolist() + df['response'].tolist()
                self.sequences = []
                for text in tqdm(texts, "Processing pre-train data"):
                    words = ['< SOS >'] + re.findall(r'\b\w+\b', text.lower()) + ['<EOS>']
                    if 3 <= len(words) <= 50:
                        indices = [word_to_idx[w] for w in words if w in word_to_idx]
                        if len(indices) >= 3:
                            self.sequences.append(torch.tensor(indices, dtype=torch.long))
            def __len__(self): return len(self.sequences)
            def __getitem__(self, idx): return self.sequences[idx]
        dataset = PretrainDataset(df, self.word_to_idx)
        def collate_fn(b): return pad_sequence(b, batch_first=True, padding_value=self.word_to_idx['<PAD>'])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def get_finetune_loader(self, df, batch_size):
        # ... (loader logic is fine, no changes needed)
        class FinetuneDataset(Dataset):
            def __init__(self, df, word_to_idx):
                self.pairs = []
                for _, r in tqdm(df.iterrows(), total=len(df), desc="Processing fine-tune data"):
                    words = ['< SOS >'] + re.findall(r'\b\w+\b', r['response'].lower()) + ['<EOS>']
                    if 3 <= len(words) <= 50 and len(r['prompt'].strip()) > 0:
                        res_idx = [word_to_idx[w] for w in words if w in word_to_idx]
                        if len(res_idx) >= 3:
                            self.pairs.append((r['prompt'], torch.tensor(res_idx, dtype=torch.long)))
            def __len__(self): return len(self.pairs)
            def __getitem__(self, idx): return self.pairs[idx]
        dataset = FinetuneDataset(df, self.word_to_idx)
        def collate_fn(b):
            p, r = zip(*b); r_pad = pad_sequence(r, batch_first=True, padding_value=self.word_to_idx['<PAD>']); return p, r_pad
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# --- Part 3: The Corrected Trainer Class ---
class LanguageModelTrainer:
    def __init__(self, nn1_frozen, nn2_trainable, embedding_layer, word_to_idx, device):
        self.nn1 = nn1_frozen
        self.nn2 = nn2_trainable
        self.embedding_layer = embedding_layer
        self.word_to_idx = word_to_idx
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    def train_phase(self, phase_name, epochs, loader, optimizer, scheduler, context_encoder=None, word_to_emb=None, accumulation_steps=4):
        print(f"\n--- Starting Phase: {phase_name} ---")
        self.nn2.train()
        criterion = nn.CrossEntropyLoss(ignore_index=self.word_to_idx['<PAD>'])
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            total_loss, total_tokens, total_correct = 0.0, 0, 0
            pbar = tqdm(loader, desc=f"{phase_name} Epoch {epoch}/{epochs}")

            for i, data in enumerate(pbar):
                optimizer.zero_grad()

                if phase_name == "Pre-training":
                    batch = data.to(self.device)
                    initial_hidden = torch.zeros(batch.size(0), 128, device=self.device)
                    inputs, targets = batch[:, :-1], batch[:, 1:]
                else: # Fine-tuning
                    prompts, responses = data
                    responses = responses.to(self.device)
                    initial_hidden = torch.stack([encode_prompt_to_context_vector(p, word_to_emb, context_encoder, self.device) for p in prompts])
                    inputs, targets = responses[:, :-1], responses[:, 1:]

                if inputs.size(1) == 0: continue

                with autocast(enabled=(self.device.type == 'cuda')):
                    # --- FIX #2: Correctly handle recurrent state and gradient flow ---
                    all_logits = []
                    hidden_state = initial_hidden

                    # Process the sequence one token at a time to build the graph
                    for t in range(inputs.size(1)):
                        current_token_embeddings = self.embedding_layer(inputs[:, t])
                        model_input = torch.cat([hidden_state, current_token_embeddings], dim=1)

                        # Get next hidden state from the frozen model.
                        # Gradients will NOT be computed for nn1's weights, but the
                        # computation graph IS maintained for hidden_state.
                        with torch.no_grad():
                            hidden_state = self.nn1(model_input)

                        # Get predictions from the trainable model.
                        # The gradient can now flow back through the hidden_state.
                        logits = self.nn2(hidden_state)
                        all_logits.append(logits)

                    # Reshape for loss calculation
                    all_logits = torch.stack(all_logits, dim=1)
                    flat_logits = all_logits.reshape(-1, all_logits.size(-1))
                    flat_targets = targets.reshape(-1)

                    loss = criterion(flat_logits, flat_targets)

                # Backward pass and optimization
                self.scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.nn2.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    # Step the scheduler AFTER the optimizer
                    if scheduler:
                        scheduler.step()

                # Calculate metrics for logging
                with torch.no_grad():
                    mask = (flat_targets != self.word_to_idx['<PAD>'])
                    total_tokens += mask.sum().item()
                    if total_tokens > 0:
                        predictions = torch.argmax(flat_logits[mask], dim=1)
                        total_correct += (predictions == flat_targets[mask]).sum().item()
                        total_loss += loss.item() * inputs.size(0) # loss is already averaged

                # Update progress bar
                avg_loss = total_loss / (i + 1) if i > 0 else loss.item()
                accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
                pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{accuracy:.2f}%", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

            checkpoint_path = os.path.join(checkpoint_dir, f'model_{phase_name.lower().replace("-", "_")}_epoch_{epoch}.pth')
            torch.save({'model_state_dict': self.nn2.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            print(f"✅ Epoch {epoch} complete. Final Loss: {avg_loss:.4f}, Final Acc: {accuracy:.2f}%. Checkpoint saved.")

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Setup paths and device ---
    embedding_file_path = '/content/Word_Embeddings (1).vec'
    fcv_model_path = '/content/FineTuned Context Vector.pth'
    rf_model_path = '/content/Reccurent FineTuning.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Running on device: {device}")

    # --- Load data and models ---
    data_processor = DataProcessor(embedding_file_path, device=device, embedding_dim=128)
    fcv_model = HierarchicalAutoencoder(input_dim=4*128, latent_dim=128).to(device)
    fcv_model.load_state_dict(torch.load(fcv_model_path, map_location=device)); fcv_model.eval(); [p.requires_grad_(False) for p in fcv_model.parameters()]
    rf_model = SymmetricalAutoencoder(input_dim=2*128, latent_dim=128).to(device)
    rf_model.load_state_dict(torch.load(rf_model_path, map_location=device))
    nn1_frozen = rf_model.encoder; nn1_frozen.eval(); [p.requires_grad_(False) for p in nn1_frozen.parameters()]
    df = pd.read_parquet("hf://datasets/Dahoas/sft-gptj-synthetic-prompt-responses/data/train-00000-of-00001-5588d64a331985b4.parquet")
    trainable_nn2 = ClassifierHead(hidden_size=128, vocab_size=data_processor.vocab_size).to(device)

    # --- User input for training setup ---
    language_model_checkpoint = input("Enter path to Classifier Head (NN2) checkpoint to resume (or press Enter to start fresh): ")
    if language_model_checkpoint and os.path.exists(language_model_checkpoint):
        print(f"Resuming training from: '{language_model_checkpoint}'")
        checkpoint = torch.load(language_model_checkpoint, map_location=device)
        trainable_nn2.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Starting training from scratch.")

    mode = input("Choose training mode ('pretrain' or 'finetune'): ").lower()
    try: epochs_to_run = int(input("Enter number of epochs for this session (e.g., 10-20): "))
    except ValueError: sys.exit("FATAL ERROR: Invalid number of epochs.")

    trainer = LanguageModelTrainer(
        nn1_frozen=nn1_frozen,
        nn2_trainable=trainable_nn2,
        embedding_layer=data_processor.embedding_layer,
        word_to_idx=data_processor.word_to_idx,
        device=device
    )

    # --- Configure Optimizer and Scheduler ---
    # --- FIX #1: Use a learning rate scheduler ---
    if mode == 'pretrain':
        loader = data_processor.get_pretrain_loader(df, batch_size=64) # Can use a larger batch size
        optimizer = optim.AdamW(trainer.nn2.parameters(), lr=1e-5, weight_decay=0.01) # Start with a placeholder LR
        total_steps = len(loader) * epochs_to_run
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps, pct_start=0.1)
        trainer.train_phase("Pre-training", epochs_to_run, loader, optimizer, scheduler, accumulation_steps=4)

    elif mode == 'finetune':
        loader = data_processor.get_finetune_loader(df, batch_size=32) # Use a smaller batch size
        optimizer = optim.AdamW(trainer.nn2.parameters(), lr=1e-5, weight_decay=0.01) # Start with a placeholder LR
        total_steps = len(loader) * epochs_to_run
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=7e-4, total_steps=total_steps, pct_start=0.1)
        trainer.train_phase("Fine-tuning", epochs_to_run, loader, optimizer, scheduler,
                            context_encoder=fcv_model, word_to_emb=data_processor.word_to_emb, accumulation_steps=4)
    else:
        print(f"Invalid mode '{mode}'. Exiting.")

    print("\n✅✅✅ Session complete! ✅✅✅")
