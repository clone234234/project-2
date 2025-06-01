import logging
logging.basicConfig(level=logging.DEBUG)
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer

def generate_text(model, vocab, start_text, max_length=50, device='cpu', temperature=1.0):
    model.eval()

    idx_2_char = {idx: char for char, idx in vocab.items()}
    vocab_size = len(vocab)
    
    print(f"Generating for: '{start_text}' (temp={temperature})")

    start_indices = []
    for char in start_text:
        if char in vocab:
            start_indices.append(vocab[char])
        else:
            print(f"Warning: '{char}' not in vocabulary, skipping")
    
    if not start_indices:
        print("No valid characters in start_text, using space")
        start_indices = [vocab.get(' ', vocab.get('<pad>', 0))]
    
    input_tensor = torch.tensor(start_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = start_text
    
    for i in range(max_length):
        with torch.no_grad():
            try:
                seq_len = input_tensor.size(1)
                src_mask = None  
                
                tgt_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                output = model(input_tensor, input_tensor, src_mask, tgt_mask)
                next_token_logits = output[:, -1, :]
                
                if temperature != 1.0 and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                k = min(5, vocab_size)  
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                
                top_k_probs = F.softmax(top_k_logits, dim=-1)
                
                valid_mask = top_k_probs > 0.01
                if valid_mask.sum() == 0:
                    next_token_idx = top_k_indices[0, 0].item()
                else:
                    if temperature > 0.7:
                        filtered_probs = top_k_probs * valid_mask.float()
                        filtered_probs = filtered_probs / filtered_probs.sum()
                        sampled_idx = torch.multinomial(filtered_probs, 1)
                        next_token_idx = top_k_indices.gather(-1, sampled_idx)[0].item()
                    else:
                        valid_indices = top_k_indices[valid_mask]
                        next_token_idx = valid_indices[0].item()
                
                if next_token_idx < 0 or next_token_idx >= vocab_size:
                    print(f"Invalid token index {next_token_idx}, stopping")
                    break
                
                if next_token_idx not in idx_2_char:
                    print(f"Token {next_token_idx} not in idx_2_char mapping, stopping")
                    break
                
                next_char = idx_2_char[next_token_idx]
                
                if len(generated_text) >= 3:
                    if generated_text[-2:] == next_char * 2:
                        print("Detected character repetition, stopping")
                        break
                
                    if len(generated_text) >= 6:
                        last_3_chars = generated_text[-3:]
                        if last_3_chars == next_char + last_3_chars[:2]:
                            print("Detected pattern repetition, stopping")
                            break
                
                generated_text += next_char
                
                next_token_tensor = torch.tensor([[next_token_idx]], device=device)
                input_tensor = torch.cat((input_tensor, next_token_tensor), dim=1)
                
                if next_char in '.!?\n' and len(generated_text) > len(start_text) + 5:
                    print(f"Found end punctuation: '{next_char}', stopping")
                    break

                if input_tensor.size(1) > 100:
                    print("Sequence too long, stopping")
                    break
                
            except Exception as e:
                print(f"Error at step {i}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    return generated_text


def test_generation(model, vocab, device):
    test_prompts = [
        ("hello", 0.3, 20), 
        ("how are", 0.5, 15),    
        ("what is", 0.6, 25),  
        ("I am", 0.4, 20)     
    ]
    
    print("\n=== TESTING FIXED GENERATION ===")
    for prompt, temp, length in test_prompts:
        print(f"\nPrompt: '{prompt}' (temp={temp}, len={length})")
        try:
            result = generate_text(model, vocab, prompt, length, device, temp)
            print(f"Result: '{result}'")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def load_model_and_vocab(device='cpu'):
    """Load saved model and vocabulary correctly"""
    try:
        checkpoint = torch.load('model.pth', map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            vocab = checkpoint['vocab']
            vocab_size = checkpoint['vocab_size']
            model_config = checkpoint.get('model_config', {})
            
            model = Transformer(
                src_vocab_size=model_config.get('src_vocab_size', vocab_size),
                tgt_vocab_size=model_config.get('tgt_vocab_size', vocab_size),
                d_model=model_config.get('d_model', 512),
                num_heads=model_config.get('num_heads', 8),
                d_ff=model_config.get('d_ff', 2048),
                num_layers=model_config.get('num_layers', 6),
                dropout=model_config.get('dropout', 0.1),
                max_seq_length=model_config.get('max_seq_length', 5000)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            print(f"Successfully loaded model and vocabulary from checkpoint")
            print(f"Vocabulary size: {vocab_size}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, vocab
        else:
            print("Loading old format model")
            model = checkpoint
            with open('vocab.txt', 'r', encoding='utf-8') as f:
                vocab = eval(f.read())
            model.to(device)
            print("Loaded saved model and vocabulary")
            return model, vocab
            
    except Exception as e:
        print(f"Could not load saved model/vocab: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_vocab_from_file(vocab_file='vocab.txt'):
    """Load vocabulary from file"""
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = eval(f.read())
        return vocab
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return {'<pad>': 0, '<unk>': 1}
    
if __name__ == '__main__':
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error initializing device: {e}")
        device = torch.device('cpu')
        print("Falling back to CPU")
    
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
        print("Using untrained model")
    
    if model is not None and vocab is not None:
        print(f"Model type: {type(model)}")
        print(f"Vocab type: {type(vocab)}")
        print(f"Vocab size: {len(vocab)}")
        test_generation(model, vocab, device)
    else:
        print("Failed to load or create model and vocabulary")