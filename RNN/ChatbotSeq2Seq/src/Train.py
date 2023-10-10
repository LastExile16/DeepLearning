import torch
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, batch, optimizer, criterion, clip, data_vocab):
    
    model.train()
    
    epoch_loss = 0
    iteration = 0
    answer_token = []
    # print(type((batch))) # only for zip type
    for src, _, trg in tqdm(batch, desc ="Training"):
        answer_token = [] # store the last answer of the batch only
        pred_token_index = []
        
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg, 1)
        
        # trg = [trg len, batch size]
        # ouput = [trg len, batch size, output dim]
        # print(f'The output of training: {output} \n Type is {type(output)} \nShape is {output.shape}')
        
        for i in range(1, len(output)):
            pred_token_index.append(output[i].argmax(1))
        for tensor_token in pred_token_index:
            answer_token.append([data_vocab.index2word[j.item()] for j in tensor_token])
                # print(f'{i}: {answer_token}')
                # raise Exception 
        
        
        output_dim = output.shape[-1]
        # print(f'BEFORE train output shape: {output.shape}')
        # print(f'BEFORE train trg shape: {trg.shape}')
        
        # trg = [(trg len), batch size]
        # output = [(trg len), batch size, output dim]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        # print(f'train output shape: {output.shape}')
        # print(f'train trg shape: {trg.shape}')
        
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        iteration += 1
        
        
    return epoch_loss / iteration, answer_token


