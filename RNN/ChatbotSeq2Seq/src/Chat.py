from src.Data import tokenizer, add_symbols

import torch

def chat(src, data_vocab, model, max_length=50):
    
    model.to(model.device)
    model.eval()
    # Tokenize the input sentence
    _, tokenized_sentence = tokenizer(src)
    # print(src)
    # print(tokenized_sentence)

    
    # Add start and end tokens and convert to tensor
    src_tensor = add_symbols(torch.tensor(data_vocab(tokenized_sentence)), data_vocab)
    src_tensor = src_tensor.unsqueeze(1).to(model.device)
    
    
    # print(src_tensor)
    
    # Forward pass through the model
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        # print(hidden.shape)
        # print(cell.shape)
        # print("________________________________")
    # Create a list to store the translated words
    trg_indexes = data_vocab(['<sos>'])
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(model.device)
    
    # Initialize variables for the decoding loop
    for _ in range(max_length):
        
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1)
        # print(f'predTokenChat: {pred_token}')
        # raise Exception
        trg_indexes.append(pred_token.item())
        
        trg_tensor = pred_token
        
        if pred_token.item() == data_vocab(['<eos>'])[0]:
            break
    # Convert the indices to words
    print((trg_indexes))
    
    answer_token = [data_vocab.index2word[i] for i in trg_indexes]
    
    # Remove the start and end tokens
    answer_token = answer_token[1:-1]
            
    print("<", ' '.join(answer_token), "\n")
