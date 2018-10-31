import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 32
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class RNN(nn.Module):
    def __init__( self, loaded_embeddings,emb_size, hidden_size, num_layers, vocab_size):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(RNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        #self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.embedding = nn.Embedding.from_pretrained(loaded_embeddings)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #self.linear = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(2*self.num_layers, batch_size, self.hidden_size)

        return hidden

    def forward(self, x, lengths):
        # reset hidden state

        batch_size, seq_len = x.size()

        self.hidden = self.init_hidden(batch_size)
        if(use_cuda):
            self.hidden = self.hidden.cuda()
        # get embedding of characters
        embed = self.embedding(x)
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), batch_first=True)
        #print(self.hidden.type())
        # fprop though RNN
        rnn_out, self.hidden = self.gru(embed, self.hidden)
        # undo packing
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # sum hidden activations of RNN across time
        rnn_out = torch.sum(rnn_out, dim=1)

        #logits = self.linear(rnn_out)
        return self.hidden.transpose(0,1).contiguous().view(batch_size, -1)

class CNN(nn.Module):
    def __init__(self,params, loaded_embeddings, emb_size, hidden_size, num_layers, vocab_size):

        super(CNN, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        #self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.embedding = nn.Embedding.from_pretrained(loaded_embeddings)
        
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size=params.kernel_size, padding=(params.kernel_size-1)//2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=params.kernel_size, padding=(params.kernel_size-1)//2)
        
        #self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        batch_size, seq_len = x.size()

        embed = self.embedding(x)
        hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
        #print(hidden.shape)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        #print(hidden.shape)
        hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
        #print(hidden.shape)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        #print(hidden.shape)
        
        hidden = torch.max(hidden, dim=1)
        #print(hidden[0].shape)
        #hidden = hidden.squeeze(dim=1)
        #print(hidden.shape)
        #logits = self.linear(hidden)
        return hidden[0]

class Classifier(nn.Module):
    def __init__( self, params, loaded_embeddings,linear_hidden_size, emb_size, rnn_hidden_size, num_layers, vocab_size):
        super(Classifier, self).__init__()
        if params.encoder == "rnn":
            self.encoder = RNN(loaded_embeddings,emb_size, rnn_hidden_size, num_layers, vocab_size)
            dim_scale = 4
        else:
            self.encoder = CNN(params, loaded_embeddings,emb_size, rnn_hidden_size, num_layers, vocab_size)
            dim_scale = 2
        self.fc1 = nn.Linear(dim_scale*rnn_hidden_size, linear_hidden_size)
        self.fc2 = nn.Linear(linear_hidden_size, 3)
    
    def forward(self, premise, premise_length, premise_idx_unsort, hypothesis, hypothesis_length, hypothesis_idx_unsort):
        premise_vector = self.encoder(premise, premise_length)
        hypothesis_vector = self.encoder(hypothesis, hypothesis_length)
        
        premise_vector = premise_vector.index_select(0,premise_idx_unsort)
        hypothesis_vector = hypothesis_vector.index_select(0,hypothesis_idx_unsort)
        #print(premise_vector.size())
        combined_vector = torch.cat([premise_vector, hypothesis_vector], dim=1)
        #print(combined_vector.size())
        x = F.relu(self.fc1(combined_vector))
        #print(x.shape)
        return self.fc2(x)