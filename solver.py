# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import Net
from utils import *

class Solver():
    def __init__(self, args):

        # load shakespeare dataset
        # bptt_len 은 학습할 때 데이터를 자르는 글자 길이

        train_iter, data_info = load_poet(args.batch_size, args.bptt_len, atomoshpere = args.atom) 

        self.vocab_size = data_info["vocab_size"]
        self.TEXT = data_info["TEXT"]

        self.net = Net(self.vocab_size, args.embed_dim,
                       args.hidden_dim, args.num_layers)
        # Loss function. Classification 문제라서 CrossEntropy를 사용한다.
        self.loss_fn = nn.CrossEntropyLoss()
        
        # optimism funciton은 Adam으로 한다.
        self.optim   = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        
        # network와 loss function을 쿠다화 시킨다.
        self.net = self.net.cuda()
        self.loss_fn = self.loss_fn.cuda()
        
        # parameter로 들어온 args와 위에서 선언한 지역변수 train_iter를 Class 내 self 변수로 바꿔준다.
        self.args = args
        self.train_iter = train_iter
        
        # ckpt_dir에 check point가 있으면 해당 경로에 directory를 생성
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        #result_dir에 result가 있으면 해당 경로에 directory를 생성
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)

    def fit(self):
        args = self.args
        
        # 한 epoch를 의미한다.
        for epoch in range(args.max_epochs):
            self.net.train()
            # 데이터를 batch로 받고 처리한다.
            for step, inputs in enumerate(self.train_iter):
                # x와 y는 각각 input과 그에 따른 결과에 대한 내용이다.
                X = inputs.text.cuda()      # [Sequence, N(Batch size)]
                y = inputs.target.cuda()    # 

                loss = 0
                
                # hidden state 때문에 또 하나의 for 문을 사용한다.
                for i in range(X.size(0)):
                    hidden = hidden if i > 0 else None
                    # 한 번에 하나의 sequence씩 input으로 넣어준다. 그리고 out과 hidden을 return 받는다.
                    out, hidden = self.net(X[i, :], hidden)          # [Batch Size, Num of classes(알파벳 개수)]
                    out = out.view(args.batch_size, -1)
                    # 각 loss 값들을 더해줘서 전체 loss를 구해준다.
                    loss += self.loss_fn(out, y[i, :])
                
                # optimize function을 사용한다.
                self.optim.zero_grad()
                
                # back-propagation을 수행한다.
                loss.backward()
                self.optim.step()
            
            # 각 epoch에 대한 sample 값들을 출력한다.
            if (epoch+1) % args.print_every == 0:
                text = self.sample(length=400)
                print("Epoch [{}/{}] loss: {:.3f}"
                    .format(epoch+1, args.max_epochs, loss.data[0]/args.bptt_len))
                print(text, "\n")
                # 파일로 저장한다.
                tem_name = ''
                if args.atom == 0:
                    tem_name = args.ckpt_name + '_lyrical'
                elif args.atom == 1:
                    tem_name = args.ckpt_name + '_love'
                elif args.atom == 2:
                    tem_name = args.ckpt_name + '_humorous'
                self.save(args.ckpt_dir, tem_name, epoch+1)
    
    # sample 함수는 sample 길이와 첫 단어를 parameter로 입력 받는다.
    def sample(self, length, prime=" "):
        self.net.eval()
        args = self.args
        
        # prime값을 torch type으로 바꿔주기 전에 prime을 list로 바꿔준다.
        samples = []
        samples.append(prime)

        # convert prime string to torch.LongTensor type
        prime = self.TEXT.process(prime, device=0, train=False).cuda()

        # prepare the first hidden state
        # hidden state를 만들어낸다.

        for i in range(prime.size(1)):
            hidden = hidden if i > 0 else None
            _, hidden = self.net(prime[:, i], hidden)
        
        # 실제 test의 첫 input은 prime의 마지막 글자로 넣어 준다.
        X = prime[:, -1]

        self.TEXT.sequential = False

        # prime의 마지막 글자의 output을 실제 RNN의 input으로 넣어준다. 그리고 각 결과를 뽑아낸다.
        for i in range(length):

            out, hidden = self.net(X, hidden)

            # sample the maximum probability character
            # return 값: max값의 score, max값의 index
            _, argmax = torch.max(out, 1)

            # preprocessing for next iteration
            # 결과 값에서 가장 큰 index를 구하고 그에 따른 값을 dictionary에서 찾아서 바꿔준다.
            out = self.TEXT.vocab.itos[argmax.data[0]]
            X = self.TEXT.numericalize([out], device=0, train=False).cuda()
            
            # 뽑아진 결과를 sample 문장에 추가한다.
            samples.append(out.replace("<eos>", "\n"))

        # <eos>를 제거한다.
        self.TEXT.sequential = True

        # samples가 list이므로 이를 string으로 바꿔준다.
        temp = "".join(samples)
        return temp

    # 정해진 path에 결과 값을 저장한다.
    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)
