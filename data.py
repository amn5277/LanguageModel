import os 
from io import open
import torch

class Dictionary(object):
    
    def __init__(self):
        self.idx2word = []
        self.word2idx = {}
        
    def addWord(self,word):
        if word not in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)+1
            return self.word2idx[word]
        
    def __len__(self):
        return len(self.idx2word)
    

class Corpus(object):
    
    def __init__(self,path):
        self.dictionary  = Dictionary()
        self.train = self.tokenize(os.path.join(path,'train.txt'))
        self.valid = self.tokenize(os.path.join(path,'valid.txt'))
        self.test = self.tokenize(os.path.join(path,'test.txt'))
        
    def tokenize(self,path):
        #Add word in the dictionary
        with open(path,'r') as f:
            for line in f:
                words  = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.addWord(word)
                    
        #Tokenize
        with open(path,'r') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids= []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        
        return ids