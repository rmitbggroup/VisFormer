import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos,i] = \
                    math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
                         requires_grad=False)
        return x
    
#The flow_attention code used here is from https://github.com/thuml/Flowformer
class FLow_Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05, eps=1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model,d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
    
    def kernel_method(self, x):
        return torch.sigmoid(x)
    
    def dot_product(self, q, k, v):
        kv = torch.einsum('nhld,nhlm->nhdm', k, v)
        qkv = torch.einsum('nhld,nhdm->nhlm', q, kv)
        return qkv
    
    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads,-1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        #Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        #Flow-Attention
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                     (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=128, dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.ones(self.size))
        self.eps = eps
    
    def forward(self,x):
        norm = self.alpha * (x - x.mean(dim=-1,keepdim=True))\
            / (x.std(dim=-1,keepdim=True)+self.eps) + self.bias
            
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attention = FLow_Attention(d_model, n_heads, dropout=0.05,eps=1e-6)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attention(x,x,x))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class NumEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1,d_model)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.linear(x)
        output = self.relu(x)
        return output
    
class ColumnEncoder(nn.Module):
    def __init__(self, d_model, N, n_heads, max_seq_len):
        super().__init__()
        self.N = N
        #self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        self.layers = get_clones(EncoderLayer(d_model, n_heads),N)
        self.norm = Norm(d_model)
    
    def forward(self, x):
        #x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)

class ColumnEncoderNum(nn.Module):
    def __init__(self, d_model, N, n_heads,max_seq_len):
        super().__init__()
        self.N = N
        #self.numencoder = NumEncoder(d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len)
        self.layers = get_clones(EncoderLayer(d_model,n_heads),N)
        self.norm = Norm(d_model)
        
    def forward(self, x):
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)
        
class FinalEncoder(nn.Module):
    def __init__(self, d_model, n_heads, N):
        super().__init__()
        self.layers = get_clones(EncoderLayer(d_model, n_heads),N)
        self.norm = Norm(d_model)
        self.N = N
    
    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)

class RelationClassifier(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model,1),
            nn.Sigmoid()
            )
    def forward(self,x):
        return self.layers(x)

class VRL(nn.Module): #Visual Relation Inference Module
    def __init__(self, vocab_size, d_model, n_heads, N, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.numencoder = NumEncoder(d_model)
        self.columnencoder = ColumnEncoder(d_model, N, n_heads, max_seq_len)
        
        self.finalencoder = FinalEncoder(d_model, n_heads, N)
        self.classifier_1 = RelationClassifier(d_model)
        self.classifier_2 = RelationClassifier(d_model)
        self.classifier_3 = RelationClassifier(d_model)
        self.classifier_4 = RelationClassifier(d_model)
        self.dummy_node = nn.Parameter(torch.randn(1,d_model))
        
    def forward(self, x, data_type, dummy_index):
        B, C, S = x.shape

        D = self.d_model
        x = x.view(B*C, S)
        y = torch.zeros(B*C, S, D).to('cuda:0')
        
        for i in range(len(data_type)):
            if data_type[i] == 1:
                y[i] = self.numencoder(x[i].unsqueeze(1))
            if data_type[i] == 2:
                y[i] = self.embed(x[i].long().unsqueeze(0))
                
        y = y.view(B*C, S, D)   
        y = self.columnencoder(y) 

        y = y.view(B, C ,S ,-1)
        y = y.mean(dim=2,keepdims=False)
        
        y = self.finalencoder(y) # B,C,D
        y = F.pad(y, [0,0,0,1,0,0])
        C += 1
        #create index for scatter operation
        dummy_index = dummy_index.unsqueeze(0) #(1, B)
        dummy_index = dummy_index.repeat(self.d_model,1).transpose(0,1).unsqueeze(1) #(B,1,D)
        #create src tensor for scatter
        src = self.dummy_node.repeat(B, 1).unsqueeze(1) #(B,1,D)
        y.scatter_(1,dummy_index,src)
        
        y1 = y.repeat(1,1,C).reshape(B,C*C,self.d_model)  
        y2 = y.repeat(1,C,1)
        X = torch.cat([y1,y2],2)
        X = X.view(B,-1, 2*self.d_model)
        
        output_1 = self.classifier_1(X).reshape(B,C,C).unsqueeze(1)
        output_2 = self.classifier_2(X).reshape(B,C,C).unsqueeze(1)
        output_3 = self.classifier_3(X).reshape(B,C,C).unsqueeze(1)
        output_4 = self.classifier_4(X).reshape(B,C,C).unsqueeze(1)
        return torch.cat([output_1,output_2,output_3,output_4],1)

class VRL_Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, N, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.numencoder = NumEncoder(d_model)
        self.columnencoder = ColumnEncoder(d_model, N, n_heads, max_seq_len)
        
        self.finalencoder = FinalEncoder(d_model, n_heads, N)
        self.classifier_1 = RelationClassifier(d_model)
        self.classifier_2 = RelationClassifier(d_model)
        self.classifier_3 = RelationClassifier(d_model)
        self.classifier_4 = RelationClassifier(d_model)
        self.dummy_node = nn.Parameter(torch.randn(1,d_model))
        
    def forward(self, x, data_type, dummy_index):
        B, C, S = x.shape
        D = self.d_model
        x = x.view(B*C, S)
        y = torch.zeros(B*C, S, D).to('cuda:0')
        for i in range(len(data_type)):
            if data_type[i] == 1:
                y[i] = self.numencoder(x[i].unsqueeze(1))
            if data_type[i] == 2:
                y[i] = self.embed(x[i].long().unsqueeze(0))
                
        y = y.view(B*C, S, D)   
        y = self.columnencoder(y)
        
        y = y.view(B, C ,S ,-1)
        y = y.mean(dim=2,keepdims=False)
        
        y = self.finalencoder(y) # B,C,D
        y = F.pad(y, [0,0,0,1,0,0])
        C += 1

        dummy_index = dummy_index.unsqueeze(0) #(1, B)
        dummy_index = dummy_index.repeat(self.d_model,1).transpose(0,1).unsqueeze(1) #(B,1,D)
        src = self.dummy_node.repeat(B, 1).unsqueeze(1) #(B,1,D)
        y.scatter_(1,dummy_index,src)
        
        return y

class CNN_Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(6144,2*d_model),
            nn.ReLU(True),
            nn.LayerNorm(2*d_model),
            
            nn.Linear(2*d_model,d_model),
            nn.ReLU(True),
            nn.LayerNorm(d_model)
        )
        
    def forward(self,x):
        B = x.shape[0]
        x = self.features(x)
        x = x.view(B,-1)
        x = self.fc(x)
        return x

    
class RelationEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            )
    def forward(self,x):
        return self.layers(x)
    
        
        

        
        

    

        