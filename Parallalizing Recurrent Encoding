import sys
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

# --- Part 1: All Prerequisite Architectures & Helper Functions ---
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
        self.decoder = nn.Sequential( # Needed for loading state dict
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

class PositionalAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PositionalAggregator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), NormResBlock(512),
            nn.Linear(512, 256), NormResBlock(256),
            nn.Linear(256, output_dim))
        self.apply(kaiming_init_weights)
    def forward(self, x): return self.network(x)

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

def get_positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float64(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return torch.from_numpy(angle_rads).float()

def encode_static(sequence_tensors, fcv_model, device):
    with torch.no_grad():
        if not sequence_tensors: return torch.zeros(128, device=device)
        current_level_tensors = [t.to(device) for t in sequence_tensors]
        while len(current_level_tensors) > 1:
            num_to_pad = (4 - (len(current_level_tensors) % 4)) % 4
            if num_to_pad > 0: current_level_tensors.extend([torch.zeros_like(current_level_tensors[0])] * num_to_pad)
            chunks = [current_level_tensors[i:i+4] for i in range(0, len(current_level_tensors), 4)]
            L_input = torch.stack([torch.cat(chunk, dim=0) for chunk in chunks])
            L_output = fcv_model.encode(L_input)
            current_level_tensors = list(L_output)
        return current_level_tensors[0]

def encode_dynamic(sequence_tensors, rf_model, initial_hidden_state, device):
    with torch.no_grad():
        sequence_tensors = [t.to(device) for t in sequence_tensors]
        if len(sequence_tensors) < 2: return []
        latent_vector = rf_model.encode(torch.cat([initial_hidden_state, sequence_tensors[0]]).unsqueeze(0))
        dynamic_latents = [latent_vector.squeeze(0)]
        for i in range(1, len(sequence_tensors) - 1):
            recurrent_input = torch.cat([latent_vector, sequence_tensors[i].unsqueeze(0)], dim=1)
            latent_vector = rf_model.encode(recurrent_input)
            dynamic_latents.append(latent_vector.squeeze(0))
        return dynamic_latents

# --- Part 2: The Memory-Efficient Custom Dataset Class ---
class NnPosDataset(Dataset):
    def __init__(self, df, word_to_emb, fcv_model, rf_model, device, is_finetune=False, max_seq_len=2048):
        self.word_to_emb = word_to_emb
        self.fcv_model = fcv_model
        self.rf_model = rf_model
        self.device = device
        self.is_finetune = is_finetune
        self.max_seq_len = max_seq_len

        self.pos_enc = get_positional_encoding(max_seq_len, 256).to(device)
        self.zeros_128 = torch.zeros(128, device=device)

        print(f"Processing data for {'Finetune' if is_finetune else 'Pretrain'} phase...")
        self.raw_data = []
        if is_finetune:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                prompt_words = re.findall(r'\b\w+\b', row['prompt'].lower())
                response_words = ['<SOS>'] + re.findall(r'\b\w+\b', row['response'].lower()) + ['<EOS>']
                if len(prompt_words) <= max_seq_len and len(response_words) <= max_seq_len:
                    self.raw_data.append({'prompt': prompt_words, 'response': response_words})
        else:
            all_texts = df['prompt'].tolist() + df['response'].tolist()
            for text in tqdm(all_texts):
                words = ['<SOS>'] + re.findall(r'\b\w+\b', text.lower()) + ['<EOS>']
                if len(words) <= max_seq_len:
                    self.raw_data.append({'text': words})

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]

        if self.is_finetune:
            prompt_tensors = [torch.from_numpy(self.word_to_emb[w]) for w in item['prompt'] if w in self.word_to_emb]
            response_tensors = [torch.from_numpy(self.word_to_emb[w]) for w in item['response'] if w in self.word_to_emb]
            if not prompt_tensors or len(response_tensors) < 2: return None

            prompt_static_context = encode_static(prompt_tensors, self.fcv_model, self.device)
            response_static_context = encode_static(response_tensors, self.fcv_model, self.device)
            dynamic_latents = encode_dynamic(response_tensors, self.rf_model, prompt_static_context, self.device)
            if not dynamic_latents: return None

            x_vecs = [torch.cat([prompt_static_context, response_static_context]) + self.pos_enc[i] for i in range(len(dynamic_latents))]
            return torch.stack(x_vecs), torch.stack(dynamic_latents)
        else:
            sequence_tensors = [torch.from_numpy(self.word_to_emb[w]) for w in item['text'] if w in self.word_to_emb]
            if len(sequence_tensors) < 2: return None

            static_context = encode_static(sequence_tensors, self.fcv_model, self.device)
            dynamic_latents = encode_dynamic(sequence_tensors, self.rf_model, self.zeros_128, self.device)
            if not dynamic_latents: return None

            x_vecs = [torch.cat([self.zeros_128, static_context]) + self.pos_enc[i] for i in range(len(dynamic_latents))]
            return torch.stack(x_vecs), torch.stack(dynamic_latents)

# --- Part 3: The Main Training Pipeline ---
def main_training_pipeline(embedding_file_path, fcv_model_path, rf_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Running on device: {device}")

    word_to_emb, embedding_dim = load_word_embeddings(embedding_file_path)
    if embedding_dim != 128: sys.exit(f"FATAL ERROR: Word embedding size must be 128.")

    special_tokens = ['<SOS>', '<EOS>']
    for token in special_tokens:
        if token not in word_to_emb:
            word_to_emb[token] = np.random.uniform(-0.1, 0.1, embedding_dim).astype(np.float32)

    print("\nLoading pre-trained models...")
    fcv_model = HierarchicalAutoencoder(input_dim=4*128, latent_dim=128).to(device)
    fcv_model.load_state_dict(torch.load(fcv_model_path, map_location=device))
    fcv_model.eval()

    rf_model = SymmetricalAutoencoder(input_dim=2*128, latent_dim=128).to(device)
    rf_model.load_state_dict(torch.load(rf_model_path, map_location=device))
    rf_model.eval()
    print("✅ All pre-trained models loaded and set to eval mode.")

    df = pd.read_parquet("hf://datasets/Dahoas/sft-gptj-synthetic-prompt-responses/data/train-00000-of-00001-5588d64a331985b4.parquet")

    nn_pos = PositionalAggregator(input_dim=256, output_dim=128).to(device)
    criterion = nn.MSELoss()
    model_checkpoint_dir = "nn_pos_checkpoints"
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    def collate_fn_nnpos(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None, None
        x_vecs, y_vecs = zip(*batch)
        return torch.cat(x_vecs, dim=0), torch.cat(y_vecs, dim=0)

    # ========================== PHASE 1: PRE-TRAINING NN_pos ==========================
    print("\n--- Starting Phase 1: Pre-training Positional Aggregator ---")
    pretrain_dataset = NnPosDataset(df, word_to_emb, fcv_model, rf_model, device, is_finetune=False)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_nnpos)
    optimizer = optim.Adam(nn_pos.parameters(), lr=1e-4)

    for epoch in range(1, 4):
        pbar = tqdm(pretrain_loader, desc=f"Pre-training Epoch {epoch}/3")
        for x_batch, y_batch in pbar:
            if x_batch is None: continue
            optimizer.zero_grad()
            preds = nn_pos(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                cosine_sim = F.cosine_similarity(preds, y_batch, dim=1).mean().item()
                accuracy = (cosine_sim + 1.0) / 2.0 * 100.0
            pbar.set_postfix({'Loss': f"{loss.item():.6f}", 'Acc': f"{accuracy:.2f}%"})
        checkpoint_path = os.path.join(model_checkpoint_dir, f'nn_pos_pretrain_epoch_{epoch}.pth')
        torch.save(nn_pos.state_dict(), checkpoint_path)
        print(f"✅ Pre-training checkpoint saved to '{checkpoint_path}'")

    # ========================== PHASE 2: FINE-TUNING NN_pos ==========================
    print("\n--- Starting Phase 2: Fine-tuning Positional Aggregator ---")
    finetune_dataset = NnPosDataset(df, word_to_emb, fcv_model, rf_model, device, is_finetune=True)
    finetune_loader = DataLoader(finetune_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_nnpos)
    optimizer = optim.Adam(nn_pos.parameters(), lr=1e-5)

    for epoch in range(1, 3):
        pbar = tqdm(finetune_loader, desc=f"Fine-tuning Epoch {epoch}/2")
        for x_batch, y_batch in pbar:
            if x_batch is None: continue
            optimizer.zero_grad()
            preds = nn_pos(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                cosine_sim = F.cosine_similarity(preds, y_batch, dim=1).mean().item()
                accuracy = (cosine_sim + 1.0) / 2.0 * 100.0
            pbar.set_postfix({'Loss': f"{loss.item():.6f}", 'Acc': f"{accuracy:.2f}%"})
        checkpoint_path = os.path.join(model_checkpoint_dir, f'nn_pos_finetune_epoch_{epoch}.pth')
        torch.save(nn_pos.state_dict(), checkpoint_path)
        print(f"✅ Fine-tuning checkpoint saved to '{checkpoint_path}'")

    print("\n✅✅✅ NN_pos training complete! ✅✅✅")


if __name__ == '__main__':
    embedding_file_path = '/content/Word_Embeddings (1).vec'
    fcv_model_path = '/content/FineTuned Context Vector.pth'
    rf_model_path = '/content/Reccurent FineTuning.pth'

    if not os.path.exists(embedding_file_path):
        dummy_emb_path = 'dummy_embeddings_128d.vec'
        print(f"'{embedding_file_path}' not found, creating dummy file.")
        dummy_vocab = ['<SOS>', '<EOS>'] + [f'word_{i}' for i in range(500)]
        with open(dummy_emb_path, 'w') as f:
            f.write(f"{len(dummy_vocab)} 128\n")
            for word in dummy_vocab: f.write(f"{word} {' '.join(map(str, np.random.uniform(-1,1,128)))}\n")
        embedding_file_path = dummy_emb_path

    if not os.path.exists(fcv_model_path):
        dummy_fcv_path = 'dummy_fcv_model.pth'
        print(f"'{fcv_model_path}' not found, creating dummy model.")
        dummy_fcv = HierarchicalAutoencoder(input_dim=4*128, latent_dim=128)
        torch.save(dummy_fcv.state_dict(), dummy_fcv_path)
        fcv_model_path = dummy_fcv_path

    if not os.path.exists(rf_model_path):
        dummy_rf_path = 'dummy_rf_model.pth'
        print(f"'{rf_model_path}' not found, creating dummy model.")
        dummy_rf = SymmetricalAutoencoder(input_dim=2*128, latent_dim=128)
        torch.save(dummy_rf.state_dict(), dummy_rf_path)
        rf_model_path = dummy_rf_path

    main_training_pipeline(embedding_file_path, fcv_model_path, rf_model_path)
