import sys
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
        try: _, embedding_dim = map(int, f.readline().split())
        except (ValueError, IndexError): sys.exit("FATAL ERROR: Invalid header in embedding file.")
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1: continue
            try: word_to_emb[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError: continue
    print(f"✅ Loaded {len(word_to_emb)} words with embedding size {embedding_dim}.")
    return word_to_emb, embedding_dim

def text_to_sequence_tensors(text, word_to_emb):
    words = re.findall(r'\b\w+\b', text.lower())
    # Return a list of numpy arrays, and the original words for validation
    sequence_tensors = [torch.from_numpy(word_to_emb[word]) for word in words if word in word_to_emb]
    valid_words = [word for word in words if word in word_to_emb]
    return sequence_tensors, valid_words

def load_and_prepare_test_data(test_file_path, word_to_emb):
    """Loads, validates, and prepares the test sentences."""
    if not os.path.exists(test_file_path):
        print(f"Warning: Test file not found at '{test_file_path}'. Skipping validation.")
        return []

    print(f"\nLoading and preparing test data from '{test_file_path}'...")
    valid_test_sequences = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sequence_tensors, valid_words = text_to_sequence_tensors(line, word_to_emb)
            # CRITICAL: Only include sentences where all words were found in the embeddings
            if len(sequence_tensors) == len(re.findall(r'\b\w+\b', line.lower())):
                if sequence_tensors: # Ensure it's not empty
                    valid_test_sequences.append(torch.stack(sequence_tensors))

    print(f"Found {len(valid_test_sequences)} valid test sentences (where all words exist in embeddings).")
    return valid_test_sequences


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
    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x): return self.encoder(x)
    def decode(self, x): return self.decoder(x)

# --- Part 3: Validation Function ---
def validate_model(model, test_sequences, criterion, device):
    model.eval() # Set model to evaluation mode
    total_test_loss = 0
    total_test_accuracy = 0
    num_sequences = len(test_sequences)

    if num_sequences == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for sequence in test_sequences:
            sequence = sequence.to(device)
            hierarchy_levels = []

            # --- Perform the same recursive encoding as in training ---
            current_level_tensors = list(sequence)
            while len(current_level_tensors) > 1:
                num_to_pad = (4 - (len(current_level_tensors) % 4)) % 4
                if num_to_pad > 0:
                    pad_vector = torch.zeros_like(current_level_tensors[0])
                    current_level_tensors.extend([pad_vector] * num_to_pad)
                chunks = [current_level_tensors[i:i+4] for i in range(0, len(current_level_tensors), 4)]
                L_input = torch.stack([torch.cat(chunk, dim=0) for chunk in chunks])
                L_output = model.encode(L_input)
                hierarchy_levels.append({'input': L_input, 'output': L_output})
                current_level_tensors = list(L_output)

            if not hierarchy_levels: continue

            # --- Calculate Multi-Scale Loss and Accuracy for validation ---
            current_sequence_loss = 0
            current_sequence_accuracy = 0
            for level in hierarchy_levels:
                reconstructed_input = model.decode(level['output'])
                current_sequence_loss += criterion(reconstructed_input, level['input']).item()
                cosine_sim = F.cosine_similarity(reconstructed_input, level['input'], dim=-1).mean().item()
                current_sequence_accuracy += (cosine_sim + 1.0) / 2.0 * 100.0

            total_test_loss += current_sequence_loss / len(hierarchy_levels)
            total_test_accuracy += current_sequence_accuracy / len(hierarchy_levels)

    return total_test_loss / num_sequences, total_test_accuracy / num_sequences

# --- Part 4: The Main Training Pipeline ---
def main_training_pipeline(embedding_file_path, model_path, test_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Running on device: {device}")

    word_to_emb, embedding_dim = load_word_embeddings(embedding_file_path)
    if embedding_dim != 128: sys.exit(f"FATAL ERROR: Word embedding size must be 128, but got {embedding_dim}.")

    print(f"\nLoading pre-trained autoencoder from '{model_path}'...")
    latent_dim = 128
    model = HierarchicalAutoencoder(input_dim=4 * embedding_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully.")

    # --- Load the Datasets ---
    test_sequences = load_and_prepare_test_data(test_file_path, word_to_emb)

    print("\nLoading and processing training dataset for fine-tuning...")
    df = pd.read_parquet("hf://datasets/Dahoas/sft-gptj-synthetic-prompt-responses/data/train-00000-of-00001-5588d64a331985b4.parquet")

    class PromptResponseDataset(Dataset):
        def __init__(self, df, word_to_emb):
            prompts = [text_to_sequence_tensors(text, word_to_emb)[0] for text in df['prompt']]
            responses = [text_to_sequence_tensors(text, word_to_emb)[0] for text in df['response']]
            self.data = [torch.stack(seq) for seq in (prompts + responses) if len(seq) > 0]
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    full_dataset = PromptResponseDataset(df, word_to_emb)
    if not full_dataset: sys.exit("\nFATAL ERROR: The processed dataset is empty.")

    def collate_fn(batch): return batch
    dataloader = DataLoader(full_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    # --- Initialize Training Components ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True)
    model_checkpoint_dir = "finetuned_progressive_checkpoints"
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    print("\n--- Starting Fine-Tuning with Progressive Interleaved Training ---")
    for epoch in range(1, 4):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/3")

        for batch_of_sequences in pbar:
            for sequence in batch_of_sequences:
                current_level_tensors = list(sequence.to(device))

                while len(current_level_tensors) > 1:
                    num_to_pad = (4 - (len(current_level_tensors) % 4)) % 4
                    if num_to_pad > 0:
                        pad_vector = torch.zeros_like(current_level_tensors[0])
                        current_level_tensors.extend([pad_vector] * num_to_pad)

                    chunks = [current_level_tensors[i:i+4] for i in range(0, len(current_level_tensors), 4)]
                    L_input = torch.stack([torch.cat(chunk, dim=0) for chunk in chunks])

                    L_output = model.encode(L_input)
                    next_level_tensors = list(L_output.detach())

                    optimizer.zero_grad()
                    reconstructed = model.decode(L_output)
                    loss = criterion(reconstructed, L_input)
                    loss.backward()
                    optimizer.step()

                    current_level_tensors = next_level_tensors

            with torch.no_grad():
                cosine_sim = F.cosine_similarity(reconstructed, L_input, dim=-1).mean().item()
                accuracy = (cosine_sim + 1.0) / 2.0 * 100.0
            pbar.set_postfix({'last_loss': f"{loss.item():.6f}", 'last_acc': f"{accuracy:.2f}%"})

        # --- Validation at the end of each epoch ---
        test_loss, test_accuracy = validate_model(model, test_sequences, criterion, device)
        print(f"Epoch {epoch}/3 Complete — Test Loss: {test_loss:.6f} — Test Accuracy: {test_accuracy:.2f}%")
        scheduler.step(test_loss)

        checkpoint_path = os.path.join(model_checkpoint_dir, f'finetuned_model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Model checkpoint saved to '{checkpoint_path}'")

    print("\n✅✅✅ Fine-tuning complete! ✅✅✅")


if __name__ == '__main__':
    embedding_file_path = 'path_to_your_128d_embeddings.vec'
    model_path = 'path_to_your_pretrained_model.pth'
    test_file_path = 'path_to_your_test_sentences.txt'

    # --- Dummy File Creation for Demonstration ---
    if not os.path.exists(embedding_file_path):
        dummy_emb_path = 'dummy_embeddings_128d.vec'
        print(f"'{embedding_file_path}' not found.")
        if not os.path.exists(dummy_emb_path):
            print(f"Creating a dummy 128d embedding file: '{dummy_emb_path}'")
            common_words = ["the", "a", "is", "in", "it", "of", "and", "to", "for", "on", "prompt", "response", "question", "answer", "explain", "what", "how", "why", "following", "code", "python", "data", "sft", "test", "sentence"]
            DUMMY_EMBED_DIM = 128
            with open(dummy_emb_path, 'w', encoding='utf-8') as f:
                f.write(f"{len(common_words)} {DUMMY_EMBED_DIM}\n")
                for word in common_words:
                    vector = ' '.join(map(str, np.random.uniform(-1, 1, DUMMY_EMBED_DIM).astype(np.float32)))
                    f.write(f"{word} {vector}\n")
            print("Dummy file created.")
        embedding_file_path = dummy_emb_path

    if not os.path.exists(model_path):
        dummy_model_path = 'dummy_model.pth'
        print(f"'{model_path}' not found.")
        if not os.path.exists(dummy_model_path):
            print(f"Creating a dummy model file: '{dummy_model_path}'")
            dummy_model = HierarchicalAutoencoder(input_dim=4*128, latent_dim=128)
            torch.save(dummy_model.state_dict(), dummy_model_path)
        model_path = dummy_model_path

    if not os.path.exists(test_file_path):
        dummy_test_path = 'dummy_test_sentences.txt'
        print(f"'{test_file_path}' not found.")
        if not os.path.exists(dummy_test_path):
            print(f"Creating a dummy test file: '{dummy_test_path}'")
            with open(dummy_test_path, 'w', encoding='utf-8') as f:
                f.write("this is a test sentence for the model\n")
                f.write("explain the python code\n")
                f.write("what is the question and answer\n")
        test_file_path = dummy_test_path

    main_training_pipeline(embedding_file_path, model_path, test_file_path)
