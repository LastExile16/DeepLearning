import torch
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, batch, criterion):
    
    model.eval()
    
    epoch_loss = 0
    iteration = 0
    with torch.no_grad():
        for src, _, trg in tqdm(batch, desc ="Evaluation"):
            
            src = src.to(device)
            trg = trg.to(device)
            
            output = model(src, trg, 0) # turn off teach forcing
            
            # trg = [trg len, batch size]
            # output = [trg_len, batch size, output dim]
            # print(f'The output of training: {output} \n Type is {type(output)} \nShape is {output.shape}')
            # Shape is torch.Size([6, 2, 63925])
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            # trg = [(trg len -1)*batch size]
            # output = [(trg len - 1) * batch size, output dim]
            try:
                loss = criterion(output, trg)
            except Exception as e:
                print(f'decoder output shape: {output.shape}')
                print(f'decoder trg shape: {trg.shape}')
                print(e)
                ## decoder output shape: torch.Size([4, 63925])
                ## decoder trg shape: torch.Size([2])
                ## Expected input batch_size (4) to match target batch_size (2).
                
            
            epoch_loss += loss.item()
            iteration += 1
            
        return epoch_loss/iteration