import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import Net
from utils import *

class Test():
    def __init__(self, prime = "시작", atom = 0, length = 400):

        self.batch_size = 128
        self.bptt_len = 30
        self.embed_dim = 500
        self.hidden_dim = 512
        self.num_layers = 3
        self.length = length

        self.ckpt_dir = "checkpoint"
        self.prime = prime
        self.atom = atom


    def run(self):

        if self.atom == 0:
            load_path = os.path.join(self.ckpt_dir, "poet_lyrical_290.pth")
        elif self.atom == 1:
            load_path = os.path.join(self.ckpt_dir, "poet_love_300.pth")
        elif self.atom == 2:
            load_path = os.path.join(self.ckpt_dir, "poet_humorous_300.pth")

        train_iter, data_info = load_poet(self.batch_size, self.bptt_len, atomoshpere = self.atom)    # ppt 는 너무 기니까 잘라주는 것임 (30개 char)
        vocab_size = data_info["vocab_size"]                                 # 알파벳 종류 개수
        TEXT = data_info["TEXT"]    

        net = Net(vocab_size, self.embed_dim,self.hidden_dim, self.num_layers) # 나머지는 hyper parameter
        net.load_state_dict(torch.load(load_path))

        net = net.cuda()

        #sampling
        net.eval()
        samples = []
        samples.append(self.prime)

        self.prime = TEXT.process(self.prime, device=0, train=False).cuda()

        for i in range(self.prime.size(1)):
            hidden = hidden if i > 0 else None
            _, hidden = net(self.prime[:, i], hidden)

        X = self.prime[:, -1] # prime의 첫글자
        #X = e를 넣었을 때 나오는 다음 값으로 바꿔주자
        TEXT.sequential = False

        for i in range(self.length): # for length 400
            out, hidden = net(X, hidden)
            _, argmax = torch.max(out, 1) # out 에 점수가 들어있다

            # preprocessing for next iteration
            out = TEXT.vocab.itos[argmax.data[0]]
            X = TEXT.numericalize([out], device=0, train=False).cuda() # 결과값으로 바꿔준다

            samples.append(out.replace("<eos>", "\n")) # 개행문자로 바꿔주는 trick

        TEXT.sequential = True

        # print(samples)

        # 시를 자연스럽게 출력.
        stend = []
        for i in range(len(samples)):
            if samples[i] == "\n":
                if i == (len(samples) - 1):
                    pass;
                elif samples[i+1] == "\n":   
                    stend.append(i)

        if len(stend) > 0:
            samples = samples[:stend[-1]]

        # <eos>로 출력되는 부분 지워준다.
        eos = [ '<', 'e', 'o', 's', '>' ]
        matching = []
        for i in range(len(samples)):
            if samples[i] in eos:
                matching.append(i)

        down = 0
        # tab = 0
        for i in matching:
            del samples[i-down]
            down += 1
            # if i == matching[-1]:
            #     tab = i-down
        # del samples[tab]

        # samples를 string형태로 바꿔준다.
        text = "".join(samples)
        text = (text + "\n")

        return text
