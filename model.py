import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


class RNNmodel(nn.Module):
    
    def __init__(self,rnnType,nToken,nInp,nHidd,nLayers,dropout= 0.5,tie_weights= False):
        
        super(RNNmodel,self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nToken,nInp)
        
        if rnnType in ['LSTM','GRU']:
            self.rnn = getattr(nn,rnnType)(nInp,nHidd,nLayers,dropout)
        self.decoder = nn.Linear(nInp,nToken)
        
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.initWeights()
        self.rnnType = rnnType
        self.nHidd = nHidd
        self.layers = nLayers
    
    def initWeights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self,inputs,hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb,hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded,hidden
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)