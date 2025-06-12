import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer

class ChainOfThought(nn.Module):
    def __init__(self, vocab_size, vocab, idx_to_char, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(ChainOfThought, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout).to(self.device)
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.idx_to_char = idx_to_char
        self.d_model = d_model

    def generate(self, question, max_length=50, temperature=1.0):
        try:
            if not isinstance(question, str):
                raise ValueError("Question must be a string")
                
            cot_prmpt = question + " Let's think step by step."
            input_idx = []
            for char in cot_prmpt:
                if char in self.vocab:
                    input_idx.append(self.vocab[char])
                else:
                    input_idx.append(self.vocab['<unk>'])
                    
            if not input_idx:
                raise ValueError("No valid tokens in input")
                
            input_tensor = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(self.device)
            generated_text = cot_prmpt
            self.eval()
            with torch.no_grad():
                for _ in range(max_length):
                    seq_len = input_tensor.size(1)
                    src_mask = None  
                    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).unsqueeze(0)
                    
                    output = self.transformer(input_tensor, input_tensor, src_mask, tgt_mask)
                    next_token_logits = output[:, -1, :]
                    
                    if temperature != 1.0 and temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    top_k_indices = torch.topk(next_token_logits, k=min(5, self.vocab_size)).indices
                    next_token_idx = top_k_indices[0, 0].item()
                    
                    generated_text += self.idx_to_char[next_token_idx]
                    input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_idx]], dtype=torch.long)], dim=1) 
                    
                    if next_token_idx == self.vocab.get('<eos>', -1):
                        break
                    
                    if len(generated_text) > len(question) + 200:  
                        break
                    if len(generated_text) > 10:
                        last_chars = generated_text[-10:]
                        if len(set(last_chars)) == 1:  
                            break
                    
            return generated_text       
        except Exception as e:
            print(f"Error in generate: {str(e)}")
            return question
    
    def forward(self, question, max_length=150, temperature=1.0):
        return self.generate(question, max_length, temperature)
    def encoder(self, text: str):
        input_idx = [self.vocab.get(char, self.vocab['<unk>']) for char in text]
        return torch.tensor(input_idx, dtype=torch.long).unsqueeze(0)
    def decoder(self, input_tensor, max_length=50, temperature=1.0):
        self.eval()
        input_tensor = input_tensor.to(self.device)
        generated_indices = input_tensor[0].tolist()

        with torch.no_grad():
            for _ in range(max_length):
                seq_len = input_tensor.size(1)
                src_mask = (input_tensor != self.vocab.get('<pad>', 0)).unsqueeze(1).unsqueeze(2)
                tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                
                output = self.transformer(input_tensor, input_tensor, src_mask, tgt_mask)
                next_token_logits = output[:, -1, :] / temperature
                next_token_idx = torch.topk(next_token_logits, k=5).indices[0, 0].item()

                generated_indices.append(next_token_idx)
                next_token_tensor = torch.tensor([[next_token_idx]], dtype=torch.long, device=self.device)
                input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
def generate_text(model, question, max_length=50, temperature=1.0):
    try:
        if not isinstance(question, str):
            raise ValueError("Question must be a string")
        
        generated_text = model.generate(question, max_length, temperature)
        return generated_text
    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        return question

def test_generation(model, vocab, device):
    test_prompts = [
        ("his mother is waiting him from home", 1, 150)
    ]
    print("\n=== TESTING FIXED GENERATION ===")
    for prompt, temp, length in test_prompts:
        print(f"\nPrompt: '{prompt}' (temp={temp}, len={length})")
        try:
            result = generate_text(model, prompt, length, temp) 
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
            
            idx_to_char = {idx: char for char, idx in vocab.items()}
            
            # Create ChainOfThought model instead of just Transformer
            cot_model = ChainOfThought(
                vocab_size=vocab_size,
                vocab=vocab,
                idx_to_char=idx_to_char,
                d_model=model_config.get('d_model', 512),
                num_heads=model_config.get('num_heads', 8),
                num_layers=model_config.get('num_layers', 6),
                d_ff=model_config.get('d_ff', 2048),
                dropout=model_config.get('dropout', 0.1)
            )
            
            cot_model.transformer.load_state_dict(checkpoint['model_state_dict'])
            cot_model.to(device)
            
            print(f"Successfully loaded model and vocabulary from checkpoint")
            print(f"Vocabulary size: {vocab_size}")
            print(f"Model parameters: {sum(p.numel() for p in cot_model.parameters()):,}")
            
            return cot_model, vocab
        else:
            print("Loading old format model")
            model = checkpoint
            with open('vocab.txt', 'r', encoding='utf-8') as f:
                vocab = eval(f.read())
            

            idx_to_char = {idx: char for char, idx in vocab.items()}
            if isinstance(model, Transformer):
                cot_model = ChainOfThought(
                    vocab_size=len(vocab),
                    vocab=vocab,
                    idx_to_char=idx_to_char
                )
                cot_model.transformer = model
                cot_model.to(device)
                return cot_model, vocab
            else:
                model.to(device)
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
    
    
    if model is not None and vocab is not None:
        print(f"Model type: {type(model)}")
        print(f"Vocab type: {type(vocab)}")
        print(f"Vocab size: {len(vocab)}")
        test_generation(model, vocab, device)
    else:
        print(" Failed to load trained model and vocabulary")
        print(" Please run train.py first to create model.pth and vocab.txt")

