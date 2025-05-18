import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer

def generate_text(model, vocab, start_text, max_length=50, device='cpu', temperature=1.0):
    model.eval()
    idx_2_char = {idx: char for char, idx in vocab.items()}
    start_indices = [vocab.get(char, vocab['<pad>']) for char in start_text]
    input_tensor = torch.tensor(start_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = start_text
    
    for _ in range(max_length):
        with torch.no_grad():
            src_mask = (input_tensor != vocab['<pad>']).unsqueeze(1).unsqueeze(2).to(device)
            try:
                output = model(input_tensor, input_tensor, src_mask)
                next_token_logits = output[:, -1, :] / temperature  
                
                
                next_token_logits = next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0]  
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
               
                if temperature > 0.7:  
                    next_token = torch.multinomial(next_token_probs, num_samples=1)
                else:  
                    next_token = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)
                
                next_token_idx = next_token[0].item()
                if next_token_idx in idx_2_char:
                    next_char = idx_2_char[next_token_idx]
                    generated_text += next_char
                    input_tensor = torch.cat((input_tensor, next_token), dim=1)
                else:
                    break
            except RuntimeError as e:
                print(f"Error during text generation: {e}")
                break
    
    return generated_text

def chat(model, vocab, device='cpu'):
    print("Chat with the transformer model (type 'exit' to quit):")
    print(" - 'temp:0.8' to set temperature (0.1-2.0, lower is more deterministic)")
    print(" - 'length:30' to set maximum generation length")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        temperature = 1.0
        max_length = 50
        
        if 'temp:' in user_input:
            try:
                temp_part = user_input.split('temp:')[1].split()[0]
                temperature = float(temp_part)
                temperature = max(0.1, min(2.0, temperature))  # Clamp between 0.1 and 2.0
                user_input = user_input.replace(f'temp:{temp_part}', '').strip()
                print(f"[Using temperature: {temperature}]")
            except:
                print("[Invalid temperature parameter, using default]")
        
        if 'length:' in user_input:
            try:
                length_part = user_input.split('length:')[1].split()[0]
                max_length = int(length_part)
                max_length = max(5, min(200, max_length))  # Clamp between 5 and 200
                user_input = user_input.replace(f'length:{length_part}', '').strip()
                print(f"[Maximum length: {max_length}]")
            except:
                print("[Invalid length parameter, using default]")
        
        try:
            response = generate_text(model, vocab, user_input, max_length=max_length, 
                                     device=device, temperature=temperature)
            print(f"Model: {response}")
        except Exception as e:
            print(f"Error: {e}")
            print("Try with different input or restart the program")

def load_model_and_vocab(model_path='model.pth', device='cpu'):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['vocab_size']
        vocab = checkpoint['vocab']
        
        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            num_layers=6,
            dropout=0.1
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("Model loaded successfully!")
        return model, vocab
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def load_vocab_from_file(file_path='input.txt'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using a basic vocabulary instead.")
        text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!-:;"
   
    special_tokens = ['<pad>']
    chars = sorted(list(set(text)))
    all_tokens = special_tokens + chars
    vocab = {char: idx for idx, char in enumerate(all_tokens)}
    
    return vocab

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, vocab = load_model_and_vocab(device=device)
    
    if model is None or vocab is None:

        vocab = load_vocab_from_file()
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")
        
  
        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            num_layers=6,
            dropout=0.1
        )
        model.to(device)
        print("Starting with untrained model")
    
    chat(model, vocab, device=device)