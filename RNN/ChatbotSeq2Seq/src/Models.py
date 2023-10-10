import random 
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, rnn_dropout, weights_matrix):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = rnn_dropout)
        
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, src):
        # print(f'encoder src shape: {src.shape}')
        ## encoder src shape: torch.Size([15, 2])
        embedded = self.dropout(self.embedding(src))
        enc_outputs, (hidden, cell_state) = self.rnn(embedded)
        # print(f'encoder outputs shape: {enc_outputs.shape}')
        # print(f'encoder hidden shape: {hidden.shape}')
        # print(f'encoder cell_state shape: {cell_state.shape}')
        ## encoder outputs shape: torch.Size([15, 2, 1])
        ## encoder hidden shape: torch.Size([1, 2, 1])
        ## encoder cell_state shape: torch.Size([1, 2, 1])
        return hidden, cell_state
        

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, rnn_dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=rnn_dropout)
        self.shapes = []
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, hidden, cell_state):
        # print("_______________________________________")
        # print(f'decoder input shape: {dec_input.shape}')
        # print(f'decoder hidden shape: {hidden.shape}')
        # print(f'decoder cell_state shape: {cell_state.shape}')
        ## _______________________________________
        ## decoder input shape: torch.Size([2])
        ## decoder hidden shape: torch.Size([1, 2, 1])
        ## decoder cell_state shape: torch.Size([1, 2, 1])
        self.shapes.append(dec_input.shape)
        old_shape = dec_input.shape
        dec_input = dec_input.unsqueeze(0)
        # print(f'after squeeze decoder input shape: {dec_input.shape}')
        ## after squeeze decoder input shape: torch.Size([1, 2])
        
        embedded = self.dropout(self.embedding(dec_input))
        # print(f'decoder embedded shape: {embedded.shape}')
        ## decoder embedded shape: torch.Size([1, 2, 1])

        # embedded = embedded.view(1, 1, -1)
        try:
            output, (hidden, cell_state) = self.rnn(embedded, (hidden, cell_state))
            # if(len(self.shapes)>(26100)):
            #     print(f'decoder dec_input shape: {dec_input.shape} compared to previous shape: {self.shapes[-1]}')
            ## decoder dec_input shape: torch.Size([1, 1]) compared to previous shape: torch.Size([1])

            
            # decoder hidden shape: torch.Size([1, 2, 1])
            # decoder cell_state shape: torch.Size([1, 2, 1])
        except Exception as e:
            print(f'decoder hidden shape: {hidden.shape}')
            print(f'decoder cell_state shape: {cell_state.shape}')
            print(f'decoder embedded shape: {embedded.shape}')
            print(f'decoder dec_input shape: {dec_input.shape} compared to previous shape: {old_shape}')
            # print(f'decoder dec_input shape: {dec_input.shape} compared to previous shape: {self.shapes[-1]}')
            print(f'list shapes: {self.shapes}')
            print(e)
            if (hidden.dim() != 3 or cell_state.dim() != 3):
                output, (hidden, cell_state) = self.rnn(embedded, (hidden.unsqueeze(0), cell_state.unsqueeze(0)))
        
        # print(f'decoder output shape: {output.shape}')
        # print(f'decoder hidden shape: {hidden.shape}')
        # print(f'decoder cell_state shape: {cell_state.shape}')
        ## decoder output shape: torch.Size([1, 2, 1])
        ## decoder hidden shape: torch.Size([1, 2, 1])
        ## decoder cell_state shape: torch.Size([1, 2, 1])
        
        # prediction = self.softmax(self.fc(output.squeeze(0)))
        prediction = self.fc_out(output.squeeze(0))
        # print(f'decoder prediction shape: {prediction.shape}')
        ## decoder prediction shape: torch.Size([2, 63925])
        
        
        return prediction, hidden, cell_state
    
     
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
         
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=1):
        
        outputs = torch.zeros(trg.size(0), trg.size(1), self.decoder.output_dim).to(self.device)
        
        # encoder_hidden = torch.zeros([1, 1, self.hidden_size]).to(device) # 1 = number of LSTM layers
        # cell_state = torch.zeros([1, 1, self.hidden_size]).to(device)  

        encoder_hidden, cell_state = self.encoder(src)
        
        # create sos token with target
        decoder_input = trg[0, :]
        # print(f'decoder_input trg[0, :]: {decoder_input}')
        # decoder_input = torch.Tensor([[0]]).long().to(device) # 0 = SOS_token
        decoder_hidden = encoder_hidden
        
        for i in range(1, len(trg)):
            
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            decoder_output, decoder_hidden, cell_state = self.decoder(decoder_input, decoder_hidden, cell_state)
            
            # place predicitons in a tensor holding predicitons for each token
            outputs[i] = decoder_output
           
        
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = decoder_output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[i] if teacher_force else top1

        return outputs
