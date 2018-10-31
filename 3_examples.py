import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
import csv
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Classifier
import time
import argparse

random.seed(134)

parser = argparse.ArgumentParser(description='SNLI 3 correct and incorrect script')
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated.")
parser.add_argument('--encoder', type=str, metavar='M',
                    help="RNN or CNN Encoder?")
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

#if not torch.cuda.is_available():
#    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 32

def load_data(filename):
    with open(filename,'r') as f:
        f_list=list(csv.reader(f,delimiter="\t"))[1:5000]
    return f_list

dict_labels={'neutral':0,'entailment':1,'contradiction':2}
dict_labels_ulta={0:'neutral',1:'entailment',2:'contradiction'}
'''def build_vocab(data):
    # Returns:
    # id2char: list of chars, where id2char[i] returns char that corresponds to char i
    # char2id: dictionary where keys represent chars and corresponding values represent indices
    # some preprocessing
    max_len1 = max([len(row[0]) for row in data])
    max_len2 = max([len(row[1]) for row in data])
    all_words= []
    for sample in data:
        all_words += sample[0]+sample[1]
    unique_words = list(set(all_words))

    id2token = unique_words
    token2id = dict(zip(unique_words, range(2,2+len(unique_words))))
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX

    return token2id, id2token, max_len1,max_len2'''

def build_vocab(data):
    # Returns:
    # id2char: list of chars, where id2char[i] returns char that corresponds to char i
    # char2id: dictionary where keys represent chars and corresponding values represent indices
    # some preprocessing
    words_to_load=100000
    PAD_IDX=words_to_load
    #UNK_IDX=words_to_load+1
    max_len1 = max([len(sent[0]) for sent in data])
    max_len2 = max([len(sent[1]) for sent in data])
    with open('wiki-news-300d-1M.vec') as f:
        loaded_embeddings_ft = np.zeros((words_to_load+1, 300))
        token2id = {}
        id2token = {}
        ordered_words_ft = []
        for i, line in enumerate(f):
            if i >= words_to_load: 
                break
            s = line.split()
            loaded_embeddings_ft[i, :] = np.asarray(s[1:])
            token2id[s[0]] = i
            id2token[i] = s[0]
            ordered_words_ft.append(s[0])
        loaded_embeddings_ft[PAD_IDX,:] = np.zeros(300)
        id2token[PAD_IDX] = '<pad>'
        token2id['<pad>'] = PAD_IDX
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
    return token2id, id2token, max_len1, max_len2, torch.from_numpy(loaded_embeddings_ft).float()

def convert_to_tokens(data):
    return [(sample[0].split(" "),sample[1].split(" "), dict_labels[sample[2]]) for sample in data]

### Function that preprocessed dataset
'''def read_data():
    #data = pkl.load(open("data.p", "rb"))
    #print(data)
    train_data, val_data, test_data = load_data('snli_train.tsv'), load_data('snli_val.tsv'), load_data('mnli_val.tsv')
    train_data, val_data, test_data = convert_to_tokens(train_data), convert_to_tokens(val_data), convert_to_tokens(test_data)
    #print(train_data)
    token2id, id2token, max_len1, max_len2 = build_vocab(train_data)
    return train_data, val_data, test_data, token2id, id2token, max_len1, max_len2'''

def read_data():
    #data = pkl.load(open("data.p", "rb"))
    #print(data)
    train_data, val_data, test_data = load_data('snli_train.tsv'), load_data('snli_val.tsv'), load_data('mnli_val.tsv')
    train_data, val_data, test_data = convert_to_tokens(train_data), convert_to_tokens(val_data), convert_to_tokens(test_data)
    #print(train_data)
    token2id, id2token, max_len1, max_len2, pretrained_vecs = build_vocab(train_data)
    return train_data, val_data, test_data, token2id, id2token, max_len1, max_len2,pretrained_vecs


train_data, val_data, test_data, token2id, id2token, MAX_SENTENCE_LENGTH_1, MAX_SENTENCE_LENGTH_2, pretrained_vecs = read_data()

#print ("Maximum sentence 1 length of dataset is {}".format(MAX_SENTENCE_LENGTH_1))
#print ("Maximum sentence 2 length of dataset is {}".format(MAX_SENTENCE_LENGTH_2))
#print ("Number of words in dataset is {}".format(len(id2token)))
#print ("Characters:")
#print (token2id.keys())

class VocabDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_tuple, token2id):
        """
        @param data_list: list of character
        @param target_list: list of targets
        """
        #print(data_tuple)
        self.sent1_list, self.sent2_list, self.target_list = zip(*data_tuple)
        #print(self.sent1_list, self.sent2_list,self.target_list)
        assert (len(self.sent1_list) == len(self.sent2_list) == len(self.target_list))
        self.token2id = token2id

    def __len__(self):
        return len(self.sent1_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        sent1_token_idx = [self.token2id[w] if w in self.token2id.keys() else UNK_IDX  for w in self.sent1_list[key][:MAX_SENTENCE_LENGTH_1]]
        sent2_token_idx = [self.token2id[w] if w in self.token2id.keys() else UNK_IDX  for w in self.sent2_list[key][:MAX_SENTENCE_LENGTH_2]]
        label = self.target_list[key]
        return [sent1_token_idx, sent2_token_idx, len(sent1_token_idx), len(sent2_token_idx), label]
    
def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    sent1_list = []
    sent2_list = []
    label_list = []
    sent1_length_list = []
    sent2_length_list=[]
    #print()
    for datum in batch:
        #print(datum)
        #time.sleep(5)
        label_list.append(datum[4])
        sent1_length_list.append(datum[2])
        sent2_length_list.append(datum[3])
    #print(label_list)

    # padding
    for datum in batch:
        padded_vec_sent1 = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_SENTENCE_LENGTH_1-datum[2])),
                                mode="constant", constant_values=0)
        sent1_list.append(padded_vec_sent1)
        padded_vec_sent2 = np.pad(np.array(datum[1]),
                                pad_width=((0,MAX_SENTENCE_LENGTH_2-datum[3])),
                                mode="constant", constant_values=0)
        sent2_list.append(padded_vec_sent2)

    _, idx_sort_1 = torch.sort(torch.tensor(sent1_length_list), dim=0, descending=True)
    _, idx_unsort_1 = torch.sort(idx_sort_1, dim=0)
    
    sent1_length_list = list(torch.tensor(sent1_length_list)[idx_sort_1])
    idx_sort_1 = Variable(idx_sort_1)
    #print(sent1_list[:3])
    sent1_list = torch.tensor(sent1_list).index_select(0,idx_sort_1)


    _, idx_sort_2 = torch.sort(torch.tensor(sent2_length_list), dim=0, descending=True)
    _, idx_unsort_2 = torch.sort(idx_sort_2, dim=0)
    
    sent2_length_list = list(torch.tensor(sent2_length_list)[idx_sort_2])
    idx_sort_2 = Variable(idx_sort_2)
    #print(len(sent2_list))
    #print(idx_sort_2.size())
    sent2_list = torch.tensor(sent2_list).index_select(0,idx_sort_2)
    
    #print(sent1_list)
    #print(sent2_list)
    #print(label_list)
    
    return [sent1_list, torch.LongTensor(sent1_length_list), idx_unsort_1, sent2_list, torch.LongTensor(sent2_length_list), idx_unsort_2, torch.LongTensor(label_list)]

train_dataset = VocabDataset(train_data, token2id)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True,pin_memory=True)

val_dataset = VocabDataset(val_data, token2id)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True,pin_memory=True)

#test_dataset = VocabDataset(test_data, token2id)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=BATCH_SIZE,
#                                           collate_fn=vocab_collate_func,
#                                           shuffle=False,pin_memory=True)

'''class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, num_classes, vocab_size):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(RNN, self).__init__()
        pretrained_vecs_tensor=torch.from_numpy(pretrained_vecs).float()
        if torch.cuda.is_available():
            pretrained_vecs_tensor=pretrained_vecs_tensor.cuda()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding.from_pretrained(pretrained_vecs_tensor)
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True) #First dimension is the batch size
        #self.linear = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers*2, batch_size, self.hidden_size)

        return hidden

    def forward(self, x, lengths):
        # reset hidden state

        batch_size, seq_len = x.size()
        if torch.cuda.is_available():
            self.hidden = self.init_hidden(batch_size).cuda()
        else:
            self.hidden = self.init_hidden(batch_size)

        # get embedding of characters
        embed = self.embedding(x)
        # pack padded sequence
        #pytorch wants sequences to be in decreasing order of lengths
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), batch_first=True)
        # fprop though RNN
        rnn_out, self.hidden = self.rnn(embed, self.hidden)
        # undo packing
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # sum hidden activations of RNN across time
        rnn_out = torch.sum(rnn_out, dim=1)

        #logits = self.linear(rnn_out)
        return self.hidden.transpose(0,1).contiguous().view(batch_size, -1)

num_classes=3
EMBED_SIZE=300
Encoder_Hidden=100

class Classifier(nn.Module):
    def __init__(self, fc_hidden_size):
        super(Classifier, self).__init__()
        self.hidden_size=fc_hidden_size
        self.linear1=nn.Linear(2*Encoder_Hidden, self.hidden_size)
        self.linear2=nn.Linear(self.hidden_size, num_classes)
        self.biRNN=RNN(emb_size=EMBED_SIZE, hidden_size=Encoder_Hidden, num_layers=1, num_classes=3, vocab_size=len(id2token))
        
    def forward(self, sent1, sent2, length1, length2, idx_unsort_1, idx_unsort_2):
        sent1_tensor = self.biRNN(sent1,length1)
        #print(sent1_tensor.shape)
        #print(idx_unsort_1.shape)
        sent1_tensor = sent1_tensor.index_select(0, idx_unsort_1)
        sent2_tensor = self.biRNN(sent2,length2)
        sent2_tensor = sent2_tensor.index_select(0, idx_unsort_2)
        combined= sent1_tensor * sent2_tensor
        #print(combined.shape)
        hidden=F.relu(self.linear1(combined))
        return self.linear2(hidden)'''
        
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    c=0
    ic=0
    model.eval()
    with torch.no_grad():
        for sent1, length1, unsort1, sent2, length2, unsort2, labels in loader:
            if torch.cuda.is_available():
                sent1_batch, length1_batch, unsort1_batch, sent2_batch, length2_batch, unsort2_batch, label_batch = sent1.cuda(), length1.cuda(), unsort1.cuda(), sent2.cuda(), length2.cuda(), unsort2.cuda(), labels.cuda()
            else:
                sent1_batch, length1_batch, unsort1_batch, sent2_batch, length2_batch, unsort2_batch, label_batch = sent1, length1, unsort1, sent2, length2, unsort2, labels

            outputs = F.softmax(model(sent1_batch, length1_batch, unsort1_batch, sent2_batch, length2_batch, unsort2_batch), dim=1)
            #print(sent1_batch[:3])
            #break
            loss = criterion(outputs, label_batch)
            predicted = outputs.max(1, keepdim=True)[1]
            this=random.randint(0,10)
            sent1_batch = sent1_batch.index_select(0, unsort1_batch)
            sent2_batch = sent2_batch.index_select(0, unsort2_batch)
            if this==5:
                for i in range(len(predicted)):
                    if c<3:
                        if predicted[i]==label_batch[i]:
                            c+=1
                            print("Correctly Classified no."+str(c))
                            premise=""
                            for id in sent1_batch[i]:
                                if id2token[id.item()]==",":
                                    premise+="<unk> "
                                elif id2token[id.item()]!="999994":
                                    premise+=id2token[id.item()]+" "
                            print("Premise:",premise)
                            hypothesis=""
                            for id in sent2_batch[i]:
                                if id2token[id.item()]==",":
                                    hypothesis+="<unk> "
                                elif id2token[id.item()]!="999994":
                                    hypothesis+=id2token[id.item()]+" "
                            print("Hypothesis: "+hypothesis)
                            print("Actual label: "+ dict_labels_ulta[label_batch[i].item()])
                            print("Predicted label: "+ dict_labels_ulta[predicted[i].item()]+"\n\n")
                    if ic<3:
                        if predicted[i]!=label_batch[i]:
                            ic+=1
                            print("Incorrectly Classified no."+str(ic))
                            premise=""
                            for id in sent1_batch[i]:
                                if id2token[id.item()]==",":
                                    premise+="<unk> "
                                elif id2token[id.item()]!="999994":
                                    premise+=id2token[id.item()]+" "
                            print("Premise:",premise)
                            hypothesis=""
                            for id in sent2_batch[i]:
                                if id2token[id.item()]==",":
                                    hypothesis+="<unk> "
                                elif id2token[id.item()]!="999994":
                                    hypothesis+=id2token[id.item()]+" "
                            print("Hypothesis: "+hypothesis)
                            print("Actual label: "+ dict_labels_ulta[label_batch[i].item()])
                            print("Predicted label: "+ dict_labels_ulta[predicted[i].item()]+"\n\n")
            if ic==3 and c==3:
                break                     
                            
            #total += labels.size(0)
            #correct.append(predicted.eq(label_batch.view_as(predicted)))
            #print("Correctly Classified thus far:\n\n ",correct)
    #return (100 * correct / total),loss

args = parser.parse_args()
state_dict = torch.load(args.model, map_location=lambda storage, loc: storage)
pretrained_vecs = pretrained_vecs.to(device)
model = Classifier(args,pretrained_vecs,linear_hidden_size= 512,emb_size=300, rnn_hidden_size=args.hidden_dim, num_layers=1,  vocab_size=len(token2id)).to(device)

model.load_state_dict(state_dict)
model.eval()
criterion = torch.nn.CrossEntropyLoss()

#list_correct,list_incorrect=
test_model(val_loader,model)
