import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils 

class AnalisisSentimen(nn.Module):
    def __init__(self, vocab_size, embed_size, num_class, document_length):
        super(AnalisisSentimen, self).__init__()
        self.embeed = nn.Embedding(num_embeddings=vocab_size, 
                        embedding_dim=embed_size)
        self.linear = nn.Linear(in_features=embed_size*document_length, out_features=num_class)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5
        self.embeed.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
    
    def forward(self, data):
        
        data = data.unsqueeze(0)
        out = self.embeed(data)
        out = torch.flatten(out, start_dim=0)
        out = self.linear(out)
        out = nn.functional.softmax(out, dim=0)
        out = out.unsqueeze(0)
       
        return out
        