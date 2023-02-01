 # -*- coding: utf-8 -*- 
from __future__ import division, print_function

import imageio  
import os 
import time 
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, models, transforms
import random

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()            
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)           
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)    
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.fc1 = nn.Linear(128,16)  
        self.fc2 = nn.Linear(16,1)  
        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):      
        # Conv. blocks     
        x = self.act1(F.max_pool2d(self.conv1(x), 3))    # 32x32x32             
        x = self.BN1(x) 
        x = self.act1(F.max_pool2d(self.conv2(x), 3))    # 64x10x10
        x = self.BN2(x) 
        x = self.act1(F.max_pool2d(self.conv3(x), 2))    # 128x4x4                

        x = self.gap(x)       
        x = x.view(x.size(0), -1)
        x = self.act2(self.fc1(x))
        x = self.fc2(x)                      

        x = torch.sigmoid(x) 

        return x
        
# Train
def train_model(model, criterion, optimizer, scheduler, num_epochs, fold, dataloaders, dataset_sizes):  
    since = time.time()
    trainLosses = []  #for loss graph
    valLosses = []
    trainAcc = [] #for acc graph
    valAcc = []

    best_acc = 0.0 # 최대정확도 저장

    for epoch in range(num_epochs):  
        print('Epoch {}/{}      '.format(epoch, num_epochs-1))  
        for phase in ['train', 'test']: 
            if phase == 'train': 
                model.train()  
            else:
                model.eval() 
            running_loss = 0.0
            running_corrects = 0
            for sample1 in dataloaders[phase]: 
                inputs1 = sample1['image']  
                labels1 = sample1['target'] 

                labels1 = labels1.to(torch.float)  
                inputs1 = inputs1.to(device) 
                labels1 = labels1.to(device) 

                optimizer.zero_grad()   

                with torch.set_grad_enabled(phase == 'train'):   
                    outputs = model(inputs1)  
                    preds = outputs.view(-1, outputs.shape[0]) 
                    preds = preds.round() 
                    outputs = outputs.squeeze(1)  
                    loss = criterion(outputs, labels1)  
                    if phase == 'train':   
                        loss.backward()  
                        optimizer.step()   

                running_loss += loss.item() * inputs1.size(0)     
                running_corrects += torch.sum(preds==labels1.data) #
            
            if phase == 'train': scheduler.step()       
            epoch_loss = running_loss / dataset_sizes[phase]            
            epoch_acc = running_corrects.double() / dataset_sizes[phase]  

            if phase =='train': 
                trainLosses.append(epoch_loss)
                trainAcc.append(epoch_acc.item())

            if phase =='test':             
                valLosses.append(epoch_loss)
                valAcc.append(epoch_acc.item())
                torch.save(model.state_dict(), os.path.join(result_dir, '{}fold-epoch-{}.pth'.format(fold, epoch)))  
                if epoch_acc > best_acc: 
                    best_acc = epoch_acc 
                    best_epoch = epoch  

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))   
    
    time_elapsed = time.time() - since
    print('Trainig complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
    print('Best val Acc: {:4f}_______Best epoch: {}\n'.format(best_acc,best_epoch)) 
    return model, trainLosses, valLosses, trainAcc, valAcc

# Data augmentation   
data_transforms = {  
    'train': transforms.Compose([     
        transforms.ToPILImage(),       
        transforms.Grayscale(num_output_channels=1),   
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
        ]),
    'test': transforms.Compose([   
        transforms.ToPILImage(), 
        transforms.Grayscale(num_output_channels=1),            
        transforms.ToTensor()
    ])
}

# Subset: different transform for Trainset/Testset
class Subset_dataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform 
    def __getitem__(self, idx):
        x = self.subset[idx]
        if self.transform:
            x['image'] = self.transform(x['image'])
        return x
    def __len__(self):
        return len(self.subset)

class RecurDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        target_frame: Path to the csv file with annotations
        root_dir: Directory with all images
        '''
        self.target_frame = pd.read_csv(csv_file) 
        self.root_dir = root_dir                 
        self.transform = transform 
    def __len__(self):
        return len(self.target_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.target_frame.iloc[idx, 0]
        img_pth = os.path.join(self.root_dir,img_name+'.tif') 
        image = imageio.imread(img_pth)
        target = self.target_frame.iloc[idx, 1]
        sample = {'name': img_name, 'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


# WeightedRandomSampler: oversampling
def weight_sampler(labels):
    _, counts = np.unique(labels, return_counts=True)             
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler



def main(args):
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)   # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if (os.path.isdir(result_dir)==False): os.mkdir(result_dir)  
    if (os.path.isdir(result_dir + '/loss')==False): os.mkdir(result_dir + '/loss') 
    if (os.path.isdir(result_dir + '/acc')==False): os.mkdir(result_dir + '/acc') 

    kf = StratifiedKFold(n_splits=5, shuffle=True)     
    # Dataset                                                             
    recur_dataset = RecurDataset(csv_file='../../datasets/csv/sorted_GESIEMENS_530.csv',root_dir='../../datasets/2D/5mm5slice/max_img')      
    recur_df = pd.read_csv('../../datasets/csv/sorted_GESIEMENS_530.csv')
 
    total_target_list = recur_df.iloc[:,1].tolist()
    num_epochs = 500
    batch_size = 32

    # test_idx, path 읽어오기
    f1 = open("../../datasets/model_idx/test_idx.txt", 'r')         
    f2 = open("../../datasets/model_idx/train_idx.txt", 'r')        
    test_lines = f1.readlines()
    train_lines = f2.readlines()
    for k,(a,b) in enumerate(zip(test_lines,train_lines)):   # 5-fold CV
        test_idx = list(map(int,a.split())) 
        train_idx = list(map(int,b.split()))

        # Trainset, Testset  
        trainset = Subset_dataset([recur_dataset[i] for i in train_idx], data_transforms['train'])
        testset = Subset_dataset([recur_dataset[i] for i in test_idx], data_transforms['test'])  
        image_datasets = {'train':trainset, 'test':testset}  # dataset
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        print('Fold: {}, train:{}, test:{}'.format(k+1, dataset_sizes['train'], dataset_sizes['test']))

        Y_train = [total_target_list[i] for i in train_idx]
        train_sampler = weight_sampler(Y_train)  
        trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size = batch_size)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size)
        dataloaders = {'train': trainloader, 'test': testloader}
        
        model = Net()               
        model = model.to(device)    
                                                                           
        criterion = nn.BCELoss()              
        criterion.to(device)                                                   
        optimizer = optim.SGD(model.parameters(), lr =0.0005, momentum = 0.5)  
        
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7) # Learning rate Decay__factor of 0.7, every 50 epochs

        # Train, Test
        model,trainLosses, valLosses, trainAcc, valAcc = train_model(model,criterion, optimizer, exp_lr_scheduler, num_epochs, k+1, dataloaders, dataset_sizes) 

    f2.close()
    f1.close()


# Train/Test settings
parser = argparse.ArgumentParser(description='NSCLC recurrence prediction----5fold CV')
parser.add_argument('--model_type', type=str, default='max', help='Single_Model_type (ex.bf max af)')
parser.add_argument('--cuda', type=str, default='0', help='cuda number')
parser.add_argument('--seed', type=int, default=42, help='set seed')
args=parser.parse_args()

result_dir = '../../datasets/results/Single_model/'+args.model_type
device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu") 

main(args)
