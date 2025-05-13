import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer

def generate_text(model, vocab, start_text, max_length=50, device='cpu'):
    model.eval()
    idx_2_char = {idx: char for char, idx in vocab.items()}
    start_indices = [vocab.get(char, vocab['<pad>']) for char in start_text]
    input_tensor = torch.tensor(start_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = start_text
    
    for _ in range(max_length):
        with torch.no_grad():
            # Create masks
            src_mask = (input_tensor != vocab['<pad>']).unsqueeze(1).unsqueeze(2)
            seq_len = input_tensor.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf')).unsqueeze(0).unsqueeze(0)
            
            # Generate next token
            output = model(input_tensor, input_tensor, src_mask, tgt_mask)
            next_token_logits = output[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            
            if next_token.item() in idx_2_char:
                next_char = idx_2_char[next_token.item()]
                generated_text += next_char
                input_tensor = torch.cat((input_tensor, next_token), dim=1)
            else:
                break
    
    return generated_text

def chat(model, vocab, device='cpu'):
    print("Chat with the transformer model (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = generate_text(model, vocab, user_input, max_length=50, device=device)
        print(f"Model: {response}")

def load_vocab_from_file(file_path='input.txt'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        # Fallback to a basic vocabulary if file not found
        print(f"Warning: {file_path} not found. Using a basic vocabulary instead.")
        text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!-:;"
    
    # Create vocabulary
    special_tokens = ['<pad>']
    chars = sorted(list(set(text)))
    all_tokens = special_tokens + chars
    vocab = {char: idx for idx, char in enumerate(all_tokens)}
    
    return vocab

if __name__ == '__main__':
    # Load vocabulary
    vocab = load_vocab_from_file()
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        dropout=0.1
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Try to load model
    try:
        checkpoint = torch.load('model.pth', map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting with untrained model")
    
    # Start chat
    chat(model, vocab, device=device)