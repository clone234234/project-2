import torch
import torch.nn as nn 
from torch.nn import functional as F
from transformer import Transformer, Encoder, Decoder

def preprocess_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]  
        lines = [line.split('\t') for line in lines]
        chars = sorted(list(set(''.join([line[0] for line in lines] + [line[1] for line in lines]))))
        vocab = {char: i for i, char in enumerate(chars)}
        vocab_size = len(vocab)
    return lines, vocab, vocab_size

def generate_mask(src, tgt, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

def train_transformer(model, data, vocab_size, num_epochs=10, batch_size=32, learning_rate=0.001, vocab=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            src = torch.tensor([[vocab[char] for char in line[0]] for line in batch_data], dtype=torch.long)
            tgt = torch.tensor([[vocab[char] for char in line[1]] for line in batch_data], dtype=torch.long)

            src_mask, tgt_mask = generate_mask(src, tgt)

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1])
            loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data)}')
def generate(model, input_seq, vocab, max_length
