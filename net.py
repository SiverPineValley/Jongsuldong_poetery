# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # nn.Module을 상속하고 이에 대한 구조체를 만든다.
    def __init__(self,vocab_size, embed_dim=300,hidden_dim=512,num_layers=3):
        super().__init__()
        # embedding을 수행한다. 문장이 들어오면 index로 바꾸어 주는데, 이를 vector 형태로 바꿔주는 역할을 한다.
        # ex) D -> 4 -> [0, 0, 0, 1, 0, 0, ...]
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # GRU를 사용한다. (input channel, output channel(hidden channel 개수), num of layers)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers)
        
        # 각 hidden state에서의 결과 값을 문자로 바꿔주는 역할을 하는 fully-connected layer.
        self.decoder = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None): # hidden state와 현재 state를 입력 값으로 받는다.
        
        # x의 첫 번째 batch size를 가져온다. index X : [N(batch size),S(sequence의 개수),C(vocabulary size)]
        batch_size = x.size(0)
        
        # embedding 수행 [N(batch size),S(Sequence),embedded_dim(-1)]
        embed = self.embedding(x)

        # encoder를 통하여 LSTM을 수행한다. input과 hidden input을 넣어주고 output과 다음에 넘겨줄 hidden을 return 한다.
        # embed.view를 통해 [S, N, embedded_dim(-1)]로 순서를 바꿔준다. 
        out, hidden = self.encoder(embed.view(1, batch_size, -1), hidden)
        
        # Fully connected layer에 out 값을 넣어준다.
        # out.view를 통해 [N,hidden_dim]으로 순서를 바꿔준다.
        out = self.decoder(out.view(out.size(0)*out.size(1), -1))
        
        return out, hidden
