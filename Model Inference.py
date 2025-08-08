import sys
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Part 1: All Prerequisite Architectures & Helper Functions ---
# We must define the exact same classes so PyTorch can load the saved models.

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

class HierarchicalAutoencoder(nn.Module): # FCV Model
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

class SymmetricalAutoencoder(nn.Module): # RF Model
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

class ClassifierHead(nn.Module): # NN2 (The final language model)
    def __init__(self, hidden_size, vocab_size):
        super(ClassifierHead, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, vocab_size))
    def forward(self, x): return self.network(x)

def load_word_embeddings(file_path):
    if not os.path.exists(file_path): sys.exit(f"FATAL ERROR: Embedding file not found at '{file_path}'")
    print(f"Loading word embeddings from '{file_path}'...")
    word_to_emb, vocab = {}, []
    with open(file_path, 'r', encoding='utf-8') as f:
        try: _, embedding_dim = map(int, f.readline().split())
        except (ValueError, IndexError): sys.exit("FATAL ERROR: Invalid header in embedding file.")
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1: continue
            try:
                word = parts[0]
                word_to_emb[word] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vocab.append(word)
            except ValueError: continue
    print(f"✅ Loaded {len(word_to_emb)} words.")
    return word_to_emb, vocab, embedding_dim

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

# --- Part 2: The Definitive Inference Function with Top-k Sampling ---

def generate_response(
    prompt_text,
    fcv_model,
    nn1_frozen,
    nn2_model,
    word_to_emb,
    word_to_idx,
    idx_to_word,
    device,
    temperature=0.8,
    top_k=50,
    max_length=100
):
    """
    Generates a response using Top-k and Temperature sampling to prevent loops.
    """
    fcv_model.eval(); nn1_frozen.eval(); nn2_model.eval()

    # 1. Encode the prompt into a single context vector
    prompt_context_vector = encode_prompt_to_context_vector(prompt_text, word_to_emb, fcv_model, device)
    hidden_state = prompt_context_vector.unsqueeze(0)

    # 2. Start the autoregressive generation with the <SOS> token
    current_token_idx = torch.tensor([word_to_idx['<SOS>']], device=device)

    generated_indices = []

    with torch.no_grad():
        for _ in range(max_length):
            current_embedding = torch.from_numpy(word_to_emb[idx_to_word[current_token_idx.item()]]).unsqueeze(0).to(device)

            nn1_input = torch.cat([hidden_state, current_embedding], dim=1)
            hidden_state = nn1_frozen(nn1_input)

            logits = nn2_model(hidden_state)

            # --- THE DEFINITIVE FIX: Top-k and Temperature Sampling ---

            # 3. Apply Temperature to the logits
            logits = logits / temperature

            # 4. Find the Top K most likely tokens
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            # 5. Convert the filtered logits to probabilities
            probabilities = F.softmax(top_k_logits, dim=-1)

            # 6. Sample from the filtered distribution
            # torch.multinomial samples an index based on the probabilities
            sampled_index_in_top_k = torch.multinomial(probabilities, num_samples=1)

            # 7. Get the actual token index from the top_k_indices
            next_token_idx = top_k_indices.gather(-1, sampled_index_in_top_k).squeeze(-1)

            # --- End of Sampling Logic ---

            if next_token_idx.item() == word_to_idx['<EOS>']:
                break

            generated_indices.append(next_token_idx.item())
            current_token_idx = next_token_idx

    generated_words = [idx_to_word[idx] for idx in generated_indices]
    return " ".join(generated_words)

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Define all three required paths ---
    embedding_file_path = '/content/Word_Embeddings (1).vec'
    fcv_model_path = '/content/FineTuned Context Vector.pth'
    rf_model_path = '/content/Reccurent FineTuning.pth'
    language_model_path = '/content/model_finetune_epoch_9.pth'

    # --- Dummy File Creation for Demonstration ---
    if not os.path.exists(embedding_file_path):
        dummy_emb_path = 'dummy_embeddings_128d.vec'
        print(f"'{embedding_file_path}' not found, creating dummy file.")
        dummy_vocab = ['<SOS>', '<EOS>', 'the', 'is', 'a', 'what', 'python', 'code', 'and', 'to', 'use', 'it', 'explain', 'following', 'language']
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

    if not os.path.exists(language_model_path):
        dummy_lm_path = 'dummy_language_model.pth'
        print(f"'{language_model_path}' not found, creating dummy model.")
        _, temp_vocab, _ = load_word_embeddings(embedding_file_path)
        temp_vocab.extend(['<SOS>', '<EOS>', '<PAD>'])
        dummy_lm = ClassifierHead(hidden_size=128, vocab_size=len(temp_vocab))
        torch.save(dummy_lm.state_dict(), dummy_lm_path)
        language_model_path = dummy_lm_path

    # --- Load all components ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Running on device: {device}")

    word_to_emb, vocab, embedding_dim = load_word_embeddings(embedding_file_path)
    special_tokens = ['<SOS>', '<EOS>', '<PAD>']
    for token in special_tokens:
        if token not in word_to_emb:
            vocab.append(token)
            word_to_emb[token] = np.random.uniform(-0.1, 0.1, embedding_dim).astype(np.float32)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    vocab_size = len(vocab)

    print("\nLoading all necessary models...")
    fcv_model = HierarchicalAutoencoder(input_dim=4*128, latent_dim=128).to(device)
    fcv_model.load_state_dict(torch.load(fcv_model_path, map_location=device))

    rf_model = SymmetricalAutoencoder(input_dim=2*128, latent_dim=128).to(device)
    rf_model.load_state_dict(torch.load(rf_model_path, map_location=device))
    nn1_frozen = rf_model.encoder

    nn2_model = ClassifierHead(hidden_size=128, vocab_size=vocab_size).to(device)
    # The saved file for the language model only contains the NN2 state dict
    nn2_model.load_state_dict(torch.load(language_model_path, map_location=device))
    print("✅ All models loaded successfully.")

    # --- Interactive Generation Loop ---
    print("\n" + "="*50)
    print("      INTERACTIVE RESPONSE GENERATOR")
    print("="*50)
    print("Enter a prompt to get a response, or type 'quit' to exit.")
    while True:
        prompt = input("\n> ")
        if prompt.lower() in ['quit', 'exit']:
            break

        response = generate_response(
            prompt_text=prompt,
            fcv_model=fcv_model,
            nn1_frozen=nn1_frozen,
            nn2_model=nn2_model,
            word_to_emb=word_to_emb,
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            device=device
        )

        print(f"Response: {response}")
