 # -*- coding: utf-8 -*- 
from __future__ import division, print_function

import os
import time
import argparse 
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, models, transforms
import random

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()            
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act4 = nn.PReLU()
        # FC layers
        self.fc1 = nn.Linear(3,16)  
        self.fc2 = nn.Linear(16,32)  
        self.fc3 = nn.Linear(32,64)  
        self.fc4 = nn.Linear(64,16)
        self.fc5 = nn.Linear(16,1)


    def forward(self, x):          
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))                
        x = self.act3(self.fc3(x)) 
        x = self.act4(self.fc4(x))
        x = self.fc5(x)           

        x = torch.sigmoid(x) 

        return x



class feature_Dataset(Dataset):
    def __init__(self, Ensemble_df, recur_df, t_idx, transform=None):   
        self.feature_frame = Ensemble_df           
        target_df = recur_df.iloc[:,1]   
        self.target_frame = target_df.loc[t_idx]      
    def __len__(self):
        return len(self.target_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.feature_frame.iloc[idx, 0:3] 
        Y = self.target_frame.iloc[idx]
        X = torch.Tensor(X)
        sample = {'feature': X, 'target': Y}
    
        return sample

def weight_sampler(labels):
    _, counts = np.unique(labels, return_counts=True)             
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# Train
def train_model(model, criterion, optimizer, scheduler, num_epochs, fold, dataloaders, dataset_sizes): 
    since = time.time()
    trainLosses = []  
    valLosses = []
    trainAcc = [] 
    valAcc = []
   
    best_acc = 0.0 

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
                inputs1 = sample1['feature'] 
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
                running_corrects += torch.sum(preds==labels1.data) 
            
            if phase == 'train': scheduler.step()        
            epoch_loss = running_loss / dataset_sizes[phase]             
            epoch_acc = running_corrects.double() / dataset_sizes[phase] 

            if phase =='train': 
                trainLosses.append(epoch_loss)
                trainAcc.append(epoch_acc.item())
           
            if phase =='test':             
                valLosses.append(epoch_loss)
                valAcc.append(epoch_acc.item())
                torch.save(model.state_dict(), os.path.join(args.ensemble_dir + '3models/'+args.model_type+'/{}fold-epoch-{}.pth'.format(fold, epoch)))   # testset의 val.accuracy/loss보고 epoch선택 필요
                if epoch_acc > best_acc: 
                    best_acc = epoch_acc 
                    best_epoch = epoch  

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))  

    time_elapsed = time.time() - since
    print('Trainig complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
    print('Best val Acc: {:4f}_______Best epoch: {}'.format(best_acc,best_epoch)) 
    print()

    return model, trainLosses, valLosses, trainAcc, valAcc
    

def main(args):  
    recur_df = pd.read_csv('./datasets/csv/sorted_GESIEMENS_530.csv')  
    f1 = open("./datasets/models_idx/noval/test_idx.txt", 'r')    
    test_lines = f1.readlines()
    idx_list = [None]*5   #train,test_idx
    num_epochs = 300
    batch_size = 32

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)   # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    '''
    epoch_A, epoch_B, epoch_C: Best epochs for single model predictions (filename)
    check model name for prediction files
    '''
    pred_df_A_list = [pd.read_csv(args.single_dir+ 'predictions_'+str(i+1)+'fold_'+args.epoch_A+'Epoch.csv', index_col='Unnamed: 0') for i in range(5)] 
    pred_df_B_list = [pd.read_csv(args.single_dir+ 'predictions_'+str(i+1)+'fold_'+args.epoch_B+'Epoch.csv', index_col='Unnamed: 0') for i in range(5)]
    pred_df_C_list = [pd.read_csv(args.single_dir+ 'predictions_'+str(i+1)+'fold_'+args.epoch_C+'Epoch.csv', index_col='Unnamed: 0') for i in range(5)]
    pred_dataset_list = [pd.concat([pred_df_A_list[j],pred_df_B_list[j],pred_df_C_list[j]], axis=1) for j in range(5)]

    ensemble_result_dir = args.ensemble_dir + "3models/"+args.model_type
    if not os.path.exists(ensemble_result_dir): os.mkdir(ensemble_result_dir)
    if not os.path.exists(ensemble_result_dir + '/loss'): os.mkdir(ensemble_result_dir + '/loss')
    if not os.path.exists(ensemble_result_dir + '/acc'): os.mkdir(ensemble_result_dir + '/acc')
    
    # New Dataset
    for j,a in enumerate(test_lines):     
        idx_list[j] = list(map(int,a.split()))       
    total_idx = [*idx_list[0],*idx_list[1],*idx_list[2],*idx_list[3],*idx_list[4]]    

    for k in range(5):
        test_idx = [*idx_list[k]]
        train_idx =  [total_idx[i] for i in range(len(total_idx)) if total_idx[i] not in test_idx]
        print(str(k+1)+'Fold')
        Ensemble_train_df = pd.concat([pred_dataset_list[t] for t in range(5) if t!=k])  
        Ensemble_test_df = pred_dataset_list[k]                                          

        trainset = feature_Dataset(Ensemble_train_df, recur_df, train_idx)
        testset = feature_Dataset(Ensemble_test_df, recur_df, test_idx)
        image_datasets = {'train':trainset, 'test':testset}  
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        print('Fold: {}, train:{}, test:{}'.format(k+1, dataset_sizes['train'], dataset_sizes['test']))
        
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size)
        dataloaders = {'train': trainloader, 'test': testloader}

        modelA = Net()               
        modelA = modelA.to(device)    
                                                                             
        criterion = nn.BCELoss()              
        criterion.to(device)                                                   
        optimizer = optim.SGD(modelA.parameters(), lr =0.01, momentum = 0.5)   

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7) # Learning rate Decay__factor of 0.7, every 50 epochs

        model,trainLosses, valLosses, trainAcc, valAcc = train_model(modelA,criterion, optimizer, exp_lr_scheduler, num_epochs, k+1, dataloaders, dataset_sizes) # exp_lr_scheduler,
        df1 = pd.DataFrame(data={"trainloss": trainLosses})  
        df2 = pd.DataFrame(data={"valloss": valLosses})
        df3 = pd.DataFrame(data={"trainacc": trainAcc})
        df4 = pd.DataFrame(data={"valacc": valAcc})
        df1.to_csv(ensemble_result_dir + "/loss/trainloss_"+str(k+1)+".csv", sep=',',index=False)   
        df2.to_csv(ensemble_result_dir + "/loss/valloss_"+str(k+1)+".csv", sep=',',index=False) 
        df3.to_csv(ensemble_result_dir + "/acc/trainacc_"+str(k+1)+".csv", sep=',',index=False)  
        df4.to_csv(ensemble_result_dir + "/acc/valacc_"+str(k+1)+".csv", sep=',',index=False) 

        print('********************************'+'\n')

    f1.close()


# Train/Test settings
parser = argparse.ArgumentParser(description='NSCLC recurrence prediction----5fold CV')
parser.add_argument('--cuda', type=str, default='0', help='cuda number')
parser.add_argument('--seed', type=int, default=42, help='seed 고정')
parser.add_argument('--model_type', type=str, default=None, help='Model_type (ex.3slices)')
parser.add_argument('--epoch_A', type=str, default=0, help='1st prediction df_epoch')
parser.add_argument('--epoch_B', type=str, default=0, help='2nd prediction df_epoch')
parser.add_argument('--epoch_C', type=str, default=0, help='3rd prediction df_epoch')
parser.add_argument('--single_dir', type=str, default='./results/Ensemble/Single_model_pred/', help='mother directory for single model predictions')   # /data2/gyeon/NSCLC
parser.add_argument('--ensemble_dir', type=str, default='./results/Ensemble/', help='directory for Ensemble model predictions')

args = parser.parse_args()
device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu") # GPU 이용 가능한지를 확인
main(args)
