# -*- coding: utf-8 -*-
import spacy
import io
import torch
import torchtext.data as data
import torchtext.datasets as datasets

class poet(datasets.LanguageModelingDataset):
    atomo = {'lyrical':0, 'love':1, 'humorous': 2}  # poetery atomosphere
    dirname = "./"
    #urls = ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"]
    #name = "tinyshakespeare"

    @classmethod
    def splits(cls,text_field,root="./data", train="lyrical_poets.txt", path = "./data/lyrical",**kwargs):
        return super(poet,cls).splits(path = path, root=root, train=train, text_field=text_field, **kwargs)

    @classmethod
    def iters(cls,batch_size=32, bptt_len=35,root="./data", repeat=False, atomoshpere = 0,**kwargs):
        TEXT = data.Field(sequential=True, tokenize=list)

        if atomoshpere == 0:
            train, = cls.splits(TEXT, root=root, train = "lyrical_poets.txt", path = "./data/lyrical", **kwargs)
        elif atomoshpere == 1:
            train, = cls.splits(TEXT, root=root, train = "love_poets.txt", path = "./data/love", **kwargs)
        elif atomoshpere == 2:
            train, = cls.splits(TEXT, root=root, train = "humorous_poets.txt", path = "./data/humorous", **kwargs)
        TEXT.build_vocab(train)

        return TEXT, data.BPTTIterator.splits((train, ), batch_size=batch_size, bptt_len=bptt_len, repeat=repeat)


def load_poet(batch_size, bptt_len, atomoshpere = 0):
    TEXT, (train_iter,) = poet.iters(batch_size=batch_size,bptt_len=bptt_len,repeat=False, atomoshpere = atomoshpere)

    data_info = {
        "vocab_size": len(TEXT.vocab),
        "TEXT": TEXT
    }

    return train_iter, data_info