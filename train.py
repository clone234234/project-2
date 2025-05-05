import torch
import torch.nn as nn 
from torch.nn import functional as F
from transformer import Transformer


def preprocess_data():
    with open('input.txt','r',encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    processed_lines = []
    for line in lines:
        filtered_line = ''.join(char for char in line if char.isalnum() or char in ' .,?!-:;')
        if filtered_line:  
            processed_lines.append(filtered_line)
    special_tokens = ['<pad>']
    chars= sorted(list(set(''.join(processed_lines))))
    all_tokens = special_tokens + chars
    vocab= {char: idx for idx, char in enumerate(all_tokens)}        
    return processed_lines, vocab, len(vocab)

def generated_mask(sz, device):
    mask = torch.triu(torch.ones((sz, sz), device= device)==1).transpose(0,1)
    mask = mask.float().masked_fill(mask==1, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def batch(batch_data, vocab, device, max_seq_length=50):

    src_seqs = [line for line in batch_data]
    tgt_seqs = [line for line in batch_data]
    if '<pad>' not in vocab:
        vocab['<pad>'] = len(vocab)
    pad_idx = vocab['<pad>']

    processed_src = []
    processed_tgt = []
    
    for src_seq, tgt_seq in zip(src_seqs, tgt_seqs):
        src_indices = [vocab.get(char, pad_idx) for char in src_seq]
        tgt_indices = [vocab.get(char, pad_idx) for char in tgt_seq]
        
        if len(src_indices) > max_seq_length:
            src_indices = src_indices[:max_seq_length]
        
        if len(tgt_indices) > max_seq_length:
            tgt_indices = tgt_indices[:max_seq_length]

        src_padding = [pad_idx] * (max_seq_length - len(src_indices))
        tgt_padding = [pad_idx] * (max_seq_length - len(tgt_indices))
        
        processed_src.append(src_indices + src_padding)
        processed_tgt.append(tgt_indices + tgt_padding)
    
    src_tensor = torch.tensor(processed_src, dtype=torch.long, device=device)
    tgt_tensor = torch.tensor(processed_tgt, dtype=torch.long, device=device)
    
    return src_tensor, tgt_tensor


def train_transformer(model, data, vocab, num_epochs=10, batch_size=32, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pad_idx = vocab['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range (0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            src_tensor, tgt_tensor = batch(batch_data, vocab, device)
            
            src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)  
            tgt_mask = (tgt_tensor != 0).unsqueeze(1).unsqueeze(2)  
            
            optimizer.zero_grad()
            output = model(src_tensor, tgt_tensor[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt_tensor[:, 1:].view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data)}')

if __name__ == "__main__":
    lines, vocab, vocab_size = preprocess_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1).to(device)
    train_transformer(model, lines, vocab, num_epochs=10, batch_size=32, device=device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    



