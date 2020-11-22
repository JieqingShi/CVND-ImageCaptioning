import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=0.3)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.init()
        
        
    def init(self):
        torch.nn.init.kaiming_uniform_(self.embedding.weight)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
        
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        captions = self.embedding(captions)
        
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        out, hidden = self.lstm(inputs)
        out = self.linear(out)
        return out
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        stopword_idx = 1
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            out = self.linear(out)  # shape is (batch_size, seq_len, vocab_size)
            word = out.argmax(dim=-1)
            caption.append(word.item())
            inputs = self.embedding(word)
            if word == stopword_idx:
                break
        return caption