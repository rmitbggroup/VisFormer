import time
import torch.nn as nn
import argparse
import os
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import model
import Data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type =int, default = 1)
parser.add_argument('--layers', type = int, default = 3)
parser.add_argument('--n_heads', type = int, default = 8)
parser.add_argument('--d_model', type = int, default = 32)
parser.add_argument('--epoch', type = int, default=60)
parser.add_argument('--NSR', type = int, default = 3)
parser.add_argument('--cols', type = int, default = 300)
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--lr', type=float,default = 1e-5)
parser.add_argument('--steps',type=int,default = 1)
parser.add_argument('--load',type=bool,default = False)
parser.add_argument('--device',type=str,default = 'cuda:0')
parser.add_argument('--data_size',type=int,default=70000)

args = parser.parse_args()

class VRL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, label, mask):
        return (((pred-label)**2)*mask).sum()/mask.sum()
    
batch_size = args.batch_size
layers = args.layers
n_heads = args.n_heads
d_model = args.d_model
epoch = args.epoch
NSR = args.NSR
cols = args.cols
max_seq_len = args.max_seq_len
lr = args.lr
steps = args.steps
load = args.load
device = args.device
data_size = args.data_size

saved_path = '../saved_model/'

if not os.path.exists(saved_path):
    os.mkdir(saved_path)
    
saved_path = saved_path + 'VRL_model batch_size {} layers {} n_heads {} d_model {} NSR {} cols {} max_seq_len {} lr {} steps {}.pth'\
    .format(batch_size, layers, n_heads, d_model, NSR, cols, max_seq_len, lr, steps)
    
path = '../data/training_data.tsv'
dataset = Data.VRLDataset(path, data_size)
dataloader = DataLoader(dataset, batch_size, 
                        collate_fn = lambda x: Data.collate_batch(x, max_seq_len, cols, NSR))

VRL_model = model.VRL(30522,d_model,n_heads,layers,max_seq_len).to(device)

if load:
    VRL_model = torch.load_state_dict(saved_path).to(device)
    
criterion = VRL_Loss()
optimizer = optim.Adam(VRL_model.parameters(), lr)

VRL_model.train()
for t in range(epoch):
    start = time.time()

    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels, masks, data_type, dummy_index = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        data_type = data_type.to(device)
        dummy_index = dummy_index.to(device)
            
        outputs = VRL_model(inputs, data_type, dummy_index)
        loss = criterion(outputs, labels, masks)
        loss = loss / steps
        loss.backward()
        running_loss += loss.item()
        if (i+1) % steps == 0:
            optimizer.step()
            VRL_model.zero_grad()
        if (i+1) % 1 == 0:
            print ('epoch: {}, batch: {}, loss: {}, time: {} seconds'.format(t+1, i+1, running_loss*steps, time.time()-start))
            
    torch.save(VRL_model.state_dict(), saved_path)
    print ('Model parameters of epoch {} have been saved'.format(t+1))