import torch
from train import TransformerModel
def generate_text(model, vocab, start_text, max_length=50, device='cpu'):
    model.eval()
    idx_to_char = {idx: char for char, idx in vocab.items()}
    
    start_indices = [vocab.get(char, vocab['<pad>']) for char in start_text]
    input_tensor = torch.tensor(start_indices, dtype=torch.long, device=device).unsqueeze(0)
    generated_text = start_text

    for _ in range(max_length):
        src_mask = (input_tensor != vocab['<pad>']).unsqueeze(1).unsqueeze(2)

        with torch.no_grad():
            output = model(input_tensor, input_tensor, src_mask, src_mask)
            next_token_logits = output[:, -1, :] 
            next_token = torch.argmax(next_token_logits, dim=-1).item()

        generated_text += idx_to_char.get(next_token, '')  
        input_tensor = torch.cat(
            [input_tensor, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1
        )

    return generated_text

def chat(model, vocab, device, max_response_length=50):
    model.eval()
    idx_to_char = {idx: char for char, idx in vocab.items()}
    print("\nChat with the Transformer (Type 'quit' to exit):")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
        
        input_indices = [vocab.get(char, vocab['<pad>']) for char in user_input]
        input_tensor = torch.tensor([input_indices], dtype=torch.long, device=device)
        src_mask = (input_tensor != vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        
        response = []
        current_input = input_tensor
        
        for _ in range(max_response_length):
            with torch.no_grad():
                output = model(current_input, current_input, src_mask, src_mask)
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
            if next_token not in idx_to_char or idx_to_char[next_token] == '<pad>':
                break
            
            response.append(idx_to_char[next_token])
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            current_input = torch.cat([current_input, next_token_tensor], dim=1)
        
        print("Bot:", ''.join(response))

if __name__ == "__main__":
    