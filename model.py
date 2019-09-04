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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_size
        self.layer = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        initial_hidden = (torch.rand(self.layer, 64, self.hidden_dim), 
                          torch.rand(self.layer, 64, self.hidden_dim))
        cap_embed = self.embed(captions[:,:-1])
        sequence_input = torch.cat((features.unsqueeze(1), cap_embed), 1)
        out, _ = self.lstm(sequence_input, initial_hidden)
        out = self.fc(out)
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass