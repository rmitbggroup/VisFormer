import Data
import os
import model
import time
import torch.nn as nn
import argparse
import pandas as pd
import PIL
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()

parser.add_argument('--layers', type = int, default = 3)
parser.add_argument('--n_heads', type = int, default = 8)
parser.add_argument('--d_model', type = int, default = 32)
parser.add_argument('--epoch', type = int, default=60)
parser.add_argument('--NSR', type = int, default = 3)
parser.add_argument('--cols', type = int, default = 300)
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--lr', type=float,default = 1e-5)
parser.add_argument('--steps',type=int,default = 1)
parser.add_argument('--device',type=str,default = 'cuda:0')
parser.add_argument('--mode',type=int,default=0)

layers = parser.parse_args().layers
n_heads = parser.parse_args().n_heads
d_model = parser.parse_args().d_model
epoch = parser.parse_args().epoch
NSR = parser.parse_args().NSR
cols = parser.parse_args().cols
max_seq_len = parser.parse_args().max_seq_len
lr = parser.parse_args().lr
steps = parser.parse_args().steps
device = parser.parse_args().device
mode = parser.parse_args().mode

d_vocab = 30522

saved_model_path = '../saved_model/'
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)
    
saved_model_path = saved_model_path + 'VPL_model layers {} n_heads {} d_model {} NSR{} cols {} max_seq_len {} steps {} mode {}.pth'.format( layers,n_heads,d_model,NSR,cols,max_seq_len,steps,mode)
    
if mode == 3 or mode == 4:
    VRL_path = '../saved_model/VRL_model layers {} n_heads {} d_model {} NSR {} cols {} max_seq_len {} lr {} steps {}.pth'.format( layers,n_heads,d_model,NSR,cols,max_seq_len,lr,steps)
    VRL_model = model.NewVRI(d_vocab,d_model,n_heads,layers,max_seq_len)
    VRL_dict = torch.load(VRI_path)
    VRL.load_state_dict(VRI_dict)
    
class VPL(nn.Module):
    def __init__(self, d_vocab, d_model, n_heads, layers, max_seq_len):
        super().__init__()
        self.vrl_encoder = model.VRL_Encoder(d_vocab, d_model, n_heads, layers, max_seq_len)
        self.cnn_encoder = model.CNN_Encoder(d_model)
        if mode == 4 or mode == 4:
            VRL_dict = VRL.state_dict()
            vrL_encoder_dict = self.vrL_encoder.state_dict()
            pretrain_dict = {k:v for k,v in VRL_dict.items() if k in vrl_encoder_dict}
            vrl_encoder_dict.update(pretrain_dict)
            self.vrl_encoder.load_state_dict(vrl_encoder_dict)
            
        if mode == 4:
            for param in self.vri_encoder.parameters():
                param.requires_grad = False
        
        self.rel_encoder_1 = model.RelationEncoder(d_model)
        self.rel_encoder_2 = model.RelationEncoder(d_model)
        self.rel_encoder_3 = model.RelationEncoder(d_model)
        self.rel_encoder_4 = model.RelationEncoder(d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(2*d_model, 4*d_model),
            nn.ReLU(),
            nn.LayerNorm(4*d_model),
            nn.Linear(4*d_model,2*d_model),
            nn.ReLU(),
            nn.LayerNorm(2*d_model),
            nn.Linear(2*d_model,1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2, data_type, vis_type, dummy_index, col_pairs):
        x1 = self.vrl_encoder(x1, data_type, dummy_index)
        x2 = self.cnn_encoder(x2)
        
        B = x2.shape[0]
        
        x1 = x1.repeat(B,1,1)
        col_index = torch.Tensor([i for i in range(x1.shape[0])])
        x11 = x1[col_index.long(),col_pairs[:,0].long()]
        x12 = x1[col_index.long(),col_pairs[:,1].long()]
        x1 = torch.cat([x11,x12],dim=1)
        
        y1 = torch.zeros(B,d_model).to(device)
        
        for i in range(B):
            if vis_type[i] == 0:
                y1[i] = self.rel_encoder_1(x1[i].unsqueeze(0)).squeeze(0)
            if vis_type[i] == 1:
                y1[i] = self.rel_encoder_2(x1[i].unsqueeze(0)).squeeze(0)
            if vis_type[i] == 2:
                y1[i] = self.rel_encoder_3(x1[i].unsqueeze(0)).squeeze(0)
            if vis_type[i] == 3:
                y1[i] = self.rel_encoder_4(x1[i].unsqueeze(0)).squeeze(0)
        
        y = torch.cat([y1,x2],dim=1)
        y = self.classifier(y)

        return y
            
if __name__ == '__main__':
    VPL_model = VPL(d_vocab, d_model, n_heads, layers, max_seq_len).to(device)
    if mode == 2:
        VPL_dict = torch.load(saved_model_path)
        VPL_model.load_state_dict(VPL_dict)
    
    file_name = '../data/vpl_dataset.tsv'
    dataset = pd.read_table(file_name,sep='\t')
    
    transform = transforms.Compose([
        transforms.Resize((50,70)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(VPL_model.parameters(),lr)
    
    for t in range(epoch):
        start = time.time()
        running_loss = 0
        
        for i, row in dataset.iterrows():
            img_info = pd.read_table(row['img_info'],sep='\t')
            table = pd.read_table(row['table_data'],sep=',')
            dummy_index = torch.Tensor(1)
            dummy_index[0] = table.shape[1]
            table, lens, data_type = Data.table_tokenizer(table, sampling_num=300, max_seq_len=512)
            table = pad_sequence(table, batch_first = True).unsqueeze(0)
            data_type_list = data_type.unsqueeze(0).view(-1,1)
            imgs = []
            col_pairs = []
            vis_type_list = []
            labels = []
            
            for img_path in img_info['img_path']:
                img = Image.open(img_path)
                img = transform(img)
                imgs.append(img)
            
            imgs = torch.stack(imgs)
            
            for vis in img_info['vis']:
                vis = eval(vis)
                vis_type_list.append(vis[2])
                col_pairs.append(torch.Tensor(vis[0:2]))
            
            for label in img_info['label']:
                labels.append(label)
            
            labels = torch.Tensor(labels).unsqueeze(1)
            vis_type_list = torch.Tensor(vis_type_list)
            col_pairs = torch.stack(col_pairs)
            
            table = table.to(device)
            imgs = imgs.to(device)
            labels = labels.to(device)
            data_type_list = data_type_list.to(device)
            vis_type_list = vis_type_list.to(device)
            dummy_index = dummy_index.to(device)
            col_pairs = col_pairs.to(device)
            outputs = VPL_model(table, imgs, data_type_list, vis_type_list, dummy_index.long(), col_pairs)
            
            loss = criterion(outputs, labels)
            loss = loss / steps
            loss.backward()
            running_loss += loss.item()
            
            if (i+1) % steps == 0:
                optimizer.step()
                VPL_model.zero_grad()
                
            if (i+1) % 100 == 0:
                print ('epoch: {}, batch: {}, loss: {}, time: {} seconds'.format(t+1, i+1, running_loss, time.time()-start))
        torch.save(VPL_model.state_dict(), saved_model_path)
        print ('Model parameters of epoch {} have been saved'.format(t+1))
            
    
    
    
    
    
    
    
    





