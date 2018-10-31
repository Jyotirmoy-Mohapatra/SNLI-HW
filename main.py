import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
from torch.autograd import Variable
import pickle as pkl
import random
import pdb
import argparse
from model import *
random.seed(134)

parser = argparse.ArgumentParser(description='PyTorch SNLI')
parser.add_argument('--path', type=str, default='', metavar='P',
                    help="path to project folder")
parser.add_argument('--store', type=str, default='output/', metavar='P',
                    help="path to store output")
parser.add_argument('--ft_home', type=str, default='./', metavar='F',
                    help="folder where fast text model is located.")
parser.add_argument('--encoder', type=str, default='rnn', metavar='M',
                    help='which model to use as encoder')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--hidden_dim', type=int, default=200, metavar='N',
                    help='size of hidden dimensions (default: 200)')
parser.add_argument('--dropout', type=float, default=0.5, metavar='D',
                    help='Dropout rate (default: 0.5)')
parser.add_argument('--weight', type=float, default=0, metavar='W',
                    help='Weight decay for adam optimizer (default: 0)')
parser.add_argument('--words_to_load', type=int, default=50000, metavar='W',
                    help='Fast Text dictionary size (default: 50000)')
parser.add_argument('--kernel_size', type=int, default=3, metavar='W',
                    help='Kernel size for CNN (default: 3)')
params = parser.parse_args()

try:
    os.mkdir(params.path+params.store[:-1])
    os.mkdir(params.path+params.store+"models")
except OSError:
    print("File Creation Error")
else:
    print("File Creation Success!")


file=open(params.path+params.store+"description.txt","w")
file.write(str(vars(params)))
file.close()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#ft_home = './'
#words_to_load = 50000
PAD_IDX = params.words_to_load
#UNK_IDX = 1
BATCH_SIZE = 32

def build_vocab(data):
    # Returns:
    # id2char: list of chars, where id2char[i] returns char that corresponds to char i
    # char2id: dictionary where keys represent chars and corresponding values represent indices
    # some preprocessing
    max_len_p = max([len(sent[0]) for sent in data])
    max_len_h = max([len(sent[1]) for sent in data])
    with open(params.ft_home + 'wiki-news-300d-1M.vec') as f:
        loaded_embeddings_ft = np.zeros((params.words_to_load+1, 300))
        word2id = {}
        id2word = {}
        ordered_words_ft = []
        for i, line in enumerate(f):
            if i >= params.words_to_load: 
                break
            s = line.split()
            loaded_embeddings_ft[i, :] = np.asarray(s[1:])
            word2id[s[0]] = i
            id2word[i] = s[0]
            ordered_words_ft.append(s[0])
        id2word[PAD_IDX] = '<pad>'
        #id2word[UNK_IDX] = '<unk>'
        word2id['<pad>'] = PAD_IDX
        #word2id['<unk>'] = UNK_IDX
        loaded_embeddings_ft[PAD_IDX] = np.zeros(300)
    """
    all_words = []
    for sent in data:
        all_words += sent[0]
        all_words += sent[1]
        
    unique_words = list(set(all_words))

    id2word = unique_words
    word2id = dict(zip(unique_words, range(2,2+len(unique_words))))
    id2word = ['<pad>', '<unk>'] + id2word
    word2id['<pad>'] = PAD_IDX
    word2id['<unk>'] = UNK_IDX
    """
    return word2id, id2word, max_len_p, max_len_h, loaded_embeddings_ft

l_dict={"neutral":0,"entailment":1,"contradiction":2}

def convert_to_words(data):
    return [(sent[0].split(" "), sent[1].split(" "),l_dict[sent[2]]) for sent in data]


def load_data(filename):
    with open(params.path+filename,'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        d = list(tsvin)
    return d[1:10000]

def read_data():
    
    train_data = load_data('snli_train.tsv')
    val_data = load_data('snli_val.tsv')
    test_data = load_data('mnli_val.tsv')
    train_data, val_data, test_data = convert_to_words(train_data), convert_to_words(val_data), convert_to_words(test_data)
    word2id, id2word, max_len_p,max_len_h,loaded_embeddings = build_vocab(train_data)
    return train_data, val_data, test_data, word2id, id2word, max_len_p, max_len_h,torch.from_numpy(loaded_embeddings).float()


train_data, val_data, test_data, word2id, id2word, max_len_p, max_len_h , loaded_embeddings = read_data()

print ("Maximum premise length of dataset is {}".format(max_len_p))
print ("Maximum hypothesis length of dataset is {}".format(max_len_h))
print ("Number of words in dataset is {}".format(len(id2word)))

loaded_embeddings = loaded_embeddings.to(device)

class VocabDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_tuple, word2id):
        """
        @param data_list: list of character
        @param target_list: list of targets

        """
        self.premise_list, self.hypothesis_list, self.target_list = zip(*data_tuple)
        assert (len(self.premise_list) == len(self.target_list))
        self.word2id = word2id

    def __len__(self):
        return len(self.premise_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        premise_word_idx = [self.word2id[c] if c in self.word2id.keys() else UNK_IDX  for c in self.premise_list[key][:max_len_p]]
        hypthesis_word_idx = [self.word2id[c] if c in self.word2id.keys() else UNK_IDX  for c in self.hypothesis_list[key][:max_len_h]]
        label = self.target_list[key]
        return [premise_word_idx, hypthesis_word_idx, len(premise_word_idx), len(hypthesis_word_idx), label]

def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    premise_list = []
    hypothesis_list = []
    label_list = []
    premise_length_list = []
    hypothesis_length_list = []

    for datum in batch:
        label_list.append(datum[4])
        premise_length_list.append(datum[2])
        hypothesis_length_list.append(datum[3])
    # padding
    for datum in batch:
        padded_vec_premise = np.pad(np.array(datum[0]),
                                pad_width=((0,max_len_p-datum[2])),
                                mode="constant", constant_values=0)
        premise_list.append(padded_vec_premise)
        
        padded_vec_hypothesis = np.pad(np.array(datum[1]),
                                pad_width=((0,max_len_h-datum[3])),
                                mode="constant", constant_values=0)
        hypothesis_list.append(padded_vec_hypothesis)
    
    premise_length_list = torch.tensor(premise_length_list)
    hypothesis_length_list = torch.tensor(hypothesis_length_list)
    
    _, premise_idx_sort = torch.sort(premise_length_list, dim=0, descending=True)
    _, premise_idx_unsort = torch.sort(premise_idx_sort, dim=0)
    
    _, hypothesis_idx_sort = torch.sort(hypothesis_length_list, dim=0, descending=True)
    _, hypothesis_idx_unsort = torch.sort(hypothesis_idx_sort, dim=0)
    
    premise_length_list = list(premise_length_list[premise_idx_sort])
    premise_idx_sort = Variable(premise_idx_sort)
    premise_idx_unsort = Variable(premise_idx_unsort)
    premise_list = torch.tensor(premise_list).index_select(0,premise_idx_sort)
    
    hypothesis_length_list = list(hypothesis_length_list[hypothesis_idx_sort])
    hypothesis_idx_sort = Variable(hypothesis_idx_sort)
    hypothesis_idx_unsort = Variable(hypothesis_idx_unsort)
    hypothesis_list = torch.tensor(hypothesis_list).index_select(0,hypothesis_idx_sort)
    
    
    return [premise_list, torch.LongTensor(premise_length_list), premise_idx_unsort, hypothesis_list, torch.LongTensor(hypothesis_length_list), hypothesis_idx_unsort,torch.LongTensor(label_list)]

# Build train, valid and test dataloaders

train_dataset = VocabDataset(train_data, word2id)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True, pin_memory=use_cuda)

val_dataset = VocabDataset(val_data, word2id)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=False, pin_memory=use_cuda)

test_dataset = VocabDataset(test_data, word2id)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=False, pin_memory=use_cuda)

val_acc_list=[]
train_acc_list=[]

val_loss_list=[]
train_loss_list=[]

model = Classifier(params,loaded_embeddings,linear_hidden_size= 512,emb_size=300, rnn_hidden_size=params.hidden_dim, num_layers=1,  vocab_size=len(id2word)).to(device)

learning_rate = 3e-4
num_epochs = params.epochs # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for premise_list, premise_length_list, premise_idx_unsort, hypothesis_list, hypothesis_length_list, hypothesis_idx_unsort,label_list in loader:
            
            if use_cuda:
                premise_list = premise_list.cuda()
                premise_length_list = premise_length_list.cuda()
                premise_idx_unsort = premise_idx_unsort.cuda()
                hypothesis_list = hypothesis_list.cuda()
                hypothesis_length_list = hypothesis_length_list.cuda()
                hypothesis_idx_unsort = hypothesis_idx_unsort.cuda()
                label_list = label_list.cuda()
            
            outputs = model(premise_list, premise_length_list, premise_idx_unsort, hypothesis_list, hypothesis_length_list, hypothesis_idx_unsort)
            loss = criterion(outputs, label_list)
            F.softmax(outputs, dim=1)
            predicted = outputs.max(1, keepdim=True)[1]

            total += label_list.size(0)
            correct += predicted.eq(label_list.view_as(predicted)).sum().item()
    return (100 * correct / total), loss



# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (premise_list, premise_length_list, premise_idx_unsort, hypothesis_list, hypothesis_length_list, hypothesis_idx_unsort,label_list) in enumerate(train_loader):
        
        if use_cuda:
            premise_list = premise_list.cuda()
            premise_length_list = premise_length_list.cuda()
            premise_idx_unsort = premise_idx_unsort.cuda()
            hypothesis_list = hypothesis_list.cuda()
            hypothesis_length_list = hypothesis_length_list.cuda()
            hypothesis_idx_unsort = hypothesis_idx_unsort.cuda()
            label_list = label_list.cuda()
            
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(premise_list, premise_length_list, premise_idx_unsort, hypothesis_list, hypothesis_length_list, hypothesis_idx_unsort)
        #print(outputs.size())
        #print(label_list.size())
        loss = criterion(outputs, label_list)
        # Backward and optimize
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 100 == 0:
            train_acc, train_loss = test_model(train_loader, model)
            # validate
            val_acc, val_loss = test_model(val_loader, model)

            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc))
        
    train_acc, train_loss = test_model(train_loader, model)
    # validate
    val_acc, val_loss = test_model(val_loader, model)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    model_file = params.path+params.store+'models/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)

print("plotting!")


import matplotlib
matplotlib.use('Agg') # no UI backend

import matplotlib.pyplot as plt

epochs=[i for i in range(1,num_epochs+1)]
plt.plot(epochs,val_acc_list,'r', label = 'validation')
plt.plot(epochs,train_acc_list,'b', label = 'training')
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.title(params.encoder+" Accuracy Convergence Plot")
plt.legend(loc='upper left')
plt.savefig(params.path+params.store+"Accuracy.png")  #savefig, don't show
plt.close()
plt.plot(epochs,val_loss_list,'r', label = 'validation')
plt.plot(epochs,train_loss_list,'b', label = 'training')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title(params.encoder+params.store+" Loss Convergence Plot")
plt.legend(loc='upper left')
plt.savefig(params.path+params.store+"Loss.png")  #savefig, don't show

print("End")