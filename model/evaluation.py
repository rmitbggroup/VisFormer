import Data
import os
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
from model import VRL
from train_vpl import VPL
import time
import torch.nn as nn
import argparse
import plotly
import os
import pandas as pd
import plotly.io as pio
import plotly.express as px
import json
import numpy as np
import csv
import argparse
import PIL
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

def create_fig(xsrc, ysrc, t, path):
    if t == 'scatter':
        plt.scatter(x=xsrc, y=ysrc)
    elif t == 'line':
        if xsrc and ysrc:
            plt.plot(xsrc, ysrc)
        elif xsrc == None:
            plt.plot(ysrc)
    elif t == 'bar':
        try:
            plt.bar(x=xsrc, height=ysrc)
        except:
            print (xsrc, ysrc)
    else:
        if xsrc and ysrc:
            plt.pie(x=xsrc, labels=ysrc)
        elif ysrc == None:
            plt.pie(x=xsrc)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path)
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--k',type=int, default=3)

k = parser.parse_args().k
device = 'cuda:0'
VRL_model = VRL(30522,32,8,3,512)
VPL_model = VPL(30522,32,8,3,512)
path1 = '../saved_model/VRL.pth'
path2 = '../saved_model/VPL.pth'
VRL_model.load_state_dict(torch.load(path1))
VPL_model.load_state_dict(torch.load(path2))

VRL_model = VRL_model.to(device)
VPL_model = VPL_model.to(device)

test_data_path = '../data/test_data.tsv'
test_data = pd.read_table(test_data_path, sep='\t').iloc[:100].reset_index(drop=True)

VRL_model.eval()
VPL_model.eval()

transform = transforms.Compose([
    transforms.Resize((50,70)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

major_type = ['scatter','line','bar','pie']
DQ = np.zeros((len(test_data),k))
OA = np.zeros((len(test_data),k))

start = time.time()
for idx, data in test_data.iterrows():
    print ('Evaluating test case {}'.format(idx))
    fid = data.fid
    table_path = data['table_data']
    table = pd.read_table(table_path,sep=',')
    num_fields = data['num_fields']
    vis_type = eval(data['types'])
    pairs = eval(data['pairs'])
    ground_truth = []
    for i, t in enumerate(vis_type):
        ground_truth.append([t]+pairs[i])
    #Prediction from VRL
    inputs, max_len, data_type = Data.table_tokenizer(table,300,512)
    dummy = torch.tensor(len(inputs)).to(device)
    data_type = data_type.unsqueeze(1).to(device)
    inputs = pad_sequence(inputs,batch_first=True).unsqueeze(0).to(device)
    
    pred = VRL_model(inputs, data_type, dummy).squeeze(0)
    #Get top-k recommended list from prediction
    pred = pred.cpu().detach().numpy()
    index = np.argsort(pred.ravel())[:-(k+1):-1]
    index = np.unravel_index(index,pred.shape)
    index = np.column_stack(index)

    pred_scores = np.zeros(k)
    #generate visualization images
    num = 0
    if not os.path.exists('./image'):
        os.mkdir('./image')
    fid = 'image/'+fid
    flag = 0
    
    for i, (t, x, y) in enumerate(index):
        if x < test_data['num_fields'][idx]:
            xsrc = list(table[table.columns[x]])
        else:
            xsrc = None
        if y < test_data['num_fields'][idx]:
            ysrc = list(table[table.columns[y]])
        else:
            ysrc = None
        vis_type = major_type[t]
        try:
            img_path = fid + str(num) + '.jpeg'
            create_fig(xsrc,ysrc,vis_type,img_path)
            num = num+1
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0).to(device)
            pred_scores[i] = VPL_model(inputs, img, data_type, torch.tensor([t]).to(device), dummy.long(), torch.tensor([x,y]).unsqueeze(0).to(device))
        except:
            pred_scores[i] = 0
            continue
        
    index = index[np.argsort(pred_scores)[::-1]]
    for i, (t,x,y) in enumerate(index):
        if [x,y] in pairs or [y,x] in pairs:
            DQ[idx][i:] = [1]*(k-i)
            break
    
    for i, (t,x,y) in enumerate(index):
        if [t,x,y] in ground_truth:
            OA[idx][i:] = [1]*(k-i)
            break
            
print (time.time()-start)
print ("Data Query Recall@(1-{}):".format(k),DQ.mean(0))
print ("Overall Recall@(1-{}):".format(k),OA.mean(0))
                
    
    
        
            
    
        
        
        
    
    
    
    
    
    
    
    
    
    



