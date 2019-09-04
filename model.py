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
        self.hidden = hidden_size
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        h0 = torch.randn(1, features.shape[0], self.hidden)
        c0 = torch.randn(1, features.shape[0], self.hidden)
        out , hidden = self.lstm(features.unsqueeze(0), (h0, c0))
        out, hidden = self.lstm(captions.unsqueeze(0), hidden)
        out = self.fc(out)
        out = out.view(-1, captions.shape[1], self.vocab_size)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass