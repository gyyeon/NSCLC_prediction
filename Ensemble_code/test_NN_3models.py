 # -*- coding: utf-8 -*- 
from __future__ import division, print_function

import os
import time 
import argparse
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, models, transforms


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


# Evaluate
def evaluate_model(test_dl, model):                                                         
    model.eval() 

    recur_prob = np.array([]) 
    predictions, actuals = list(), list()         
    for sample in test_dl: 
        inputs = sample['feature']
        targets = sample['target']
        inputs = inputs.to(device) 
        targets = targets.to(device) 
        output = model(inputs)  
        output = output.detach().cpu().numpy()        
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 1))
        recur_prob = np.append(recur_prob,output) 
        output_r = output.round()       
  
        predictions.append(output_r)  
        actuals.append(actual)      
    auc = roc_auc_score(np.vstack(actuals), np.vstack(recur_prob)) 
    scores = precision_recall_fscore_support(np.vstack(actuals), np.vstack(predictions), average='binary') 
    acc = accuracy_score(np.vstack(actuals), np.vstack(predictions)) 
    predictions = sum([list(map(int, i)) for i in predictions], [])  
    actuals = sum([list(map(int, i)) for i in actuals], [])

    return auc, acc, predictions, actuals, recur_prob, scores


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
        X = self.feature_frame.iloc[idx, 0:3]   # 3models:A,B,C
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

def main(args):  
    '''
    epoch_A, epoch_B, epoch_C: Best epochs for single model predictions
    check model name for prediction files
    '''
    pred_df_A_list = [pd.read_csv(single_dir + str(args.single_type[0])+'/'+str(i+1)+'fold_'+args.epoch_A+'Epoch.csv', index_col='Unnamed: 0') for i in range(5)] 
    pred_df_B_list = [pd.read_csv(single_dir + str(args.single_type[1])+'/'+str(i+1)+'fold_'+args.epoch_B+'Epoch.csv', index_col='Unnamed: 0') for i in range(5)]
    pred_df_C_list = [pd.read_csv(single_dir + str(args.single_type[2])+'/'+str(i+1)+'fold_'+args.epoch_C+'Epoch.csv', index_col='Unnamed: 0') for i in range(5)]
    pred_dataset_list = [pd.concat([pred_df_A_list[j],pred_df_B_list[j],pred_df_C_list[j]], axis=1) for j in range(5)]

    if (os.path.isdir(ensemble_result_dir)==False): os.mkdir(ensemble_result_dir) 
    if (os.path.isdir(ensemble_result_dir+'/recur_prob')==False): os.mkdir(ensemble_result_dir+'/recur_prob') 
    if (os.path.isdir(ensemble_result_dir+'/recur_pred')==False): os.mkdir(ensemble_result_dir+'/recur_pred') 

    recur_df = pd.read_csv('../../datasets/csv/sorted_GESIEMENS_530.csv')   
    total_name_list = recur_df.iloc[:,0].tolist()   
    batch_size = 32
    best_acc = 0

    fold1_test_acc = []
    fold2_test_acc = []
    fold3_test_acc = []
    fold4_test_acc = []
    fold5_test_acc = []
    mean_test_acc = [] 
    load_idx = []
    auc_score = []

    for load in range(0,300):      
        load_idx.append('epoch_'+str(load))
        total_accuracy, total_auc = 0.0, 0.0

        f1 = open("../../datasets/model_idx/test_idx.txt", 'r')  
        lines = f1.readlines()

        for j,a in enumerate(lines):   # 5-fold  
            test_idx = list(map(int,a.split())) 
            test_fn = [total_name_list[i] for i in test_idx]   

            Ensemble_test_df = pred_dataset_list[j]     
            testset = feature_Dataset(Ensemble_test_df, recur_df, test_idx)

            # Model
            modelA = Net()             
            modelA = modelA.to(device)   
                                              
            # Dataloader    
            testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size)
            dataloaders = {'test': testloader}    

            # Test
            modelA.load_state_dict(torch.load(os.path.join(ensemble_result_dir,'/{}fold-epoch-{}.pth'.format(j+1,load))),strict=False)
            auc, acc, predictions, actuals, recur_prob, scores = evaluate_model(dataloaders['test'], modelA)  
            print(str(j+1) + 'fold____Accuracy: %.3f' % acc) 

            if j==0: fold1_test_acc.append(acc)         
            elif j==1: fold2_test_acc.append(acc)
            elif j==2: fold3_test_acc.append(acc)  
            elif j==3: fold4_test_acc.append(acc)
            else: fold5_test_acc.append(acc)

            total_accuracy += acc
            total_auc += auc
            mean_accuracy = total_accuracy/5.0
            mean_auc = total_auc/5.0

        if mean_accuracy>=best_acc: 
            best_acc = mean_accuracy
            best_epoch = load
        print('5 fold CV Accuracy_______%depoch : %.3f' % (load,mean_accuracy))
        print('Best Accuracy__________%depoch: %.3f' %(best_epoch, best_acc))
        mean_test_acc.append(mean_accuracy)

    df = pd.DataFrame(data={"epoch":load_idx, "eval_acc": mean_test_acc})  
    best_epoch_list = [i for i in df.index.tolist() if df.loc[i,'eval_acc']==best_acc]
    

    for epoch in best_epoch_list:
        f3 = open(ensemble_result_dir + "/NN_metrics_"+str(epoch)+"Epoch.txt", 'w')
        precision_score = []
        f1_score = []
        recall_score = []
        auc_score = []
        acc_list = []

        print('====================Epoch '+str(epoch)+'====================')
        for k,a in enumerate(lines):   # 5fold   
            test_idx = list(map(int,a.split())) 
            test_fn = [total_name_list[i] for i in test_idx]  

            Ensemble_test_df = pred_dataset_list[k]    
            testset = feature_Dataset(Ensemble_test_df, recur_df, test_idx)

            modelA = Net()          
            modelA = modelA.to(device)   

            testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size)

            modelA.load_state_dict(torch.load(os.path.join(ensemble_result_dir + '/{}fold-epoch-{}.pth'.format(k+1,epoch))), strict=False)
            auc, acc, predictions, actuals, recur_prob, scores = evaluate_model(testloader, modelA)  #
            precision_score.append(scores[0])  
            recall_score.append(scores[1])
            f1_score.append(scores[2])
            auc_score.append(auc)     
            acc_list.append(acc)     
            f3.write(str(classification_report(actuals, predictions))+'\n') 

            # prob_data = pd.DataFrame(data={"recur_prob": recur_prob}, index = test_fn)  
            pred_data = pd.DataFrame(data={"predictions": predictions}, index = test_fn)       
            pred_data.to_csv(ensemble_result_dir + '/recur_pred/'+str(k+1)+'fold_'+str(epoch)+'Epoch.csv') 
        print('Acc: '+str(acc_list)+'-----'+str(sum(acc_list)/5))
        print('AUC: '+str(sum(auc_score)/5)+'\n')
        print('F1 score: '+str(f1_score)+'-----'+str(sum(f1_score)/5))
        print('precision: '+str(precision_score)+'-----'+str(sum(precision_score)/5))
        print('recall: '+str(recall_score)+'-----'+str(sum(recall_score)/5))
        f3.write('Acc: '+str(acc_list)+'---'+str(sum(acc_list)/5.0)+'\nAUC :'+str(auc_score)+'---'+ str(sum(auc_score)/5)
                +'\n\nF1 : '+str(f1_score)+'---'+ str(sum(f1_score)/5)+'\n\nPrecision : '+str(precision_score)+'---'+ str(sum(precision_score)/5)
                +'\n\nRecall : '+str(recall_score)+'---'+ str(sum(recall_score)/5))
        f3.close()

    f1.close()

# Train/Test settings
parser = argparse.ArgumentParser(description='NSCLC recurrence prediction----5fold CV')
parser.add_argument('--cuda', type=str, default='0', help='cuda number')
parser.add_argument('--seed', type=int, default=42, help='set seed')
parser.add_argument('--ensemble_type', type=str, default=None, help='Ensemble_Model_type (ex.3slices)')
parser.add_argument('--single_type', nargs="+", default=["bf", "max","af"], help='Single_Model_type (ex.bf max af)')
parser.add_argument('--epoch_A', type=str, default=0, help='1st prediction best_epoch')
parser.add_argument('--epoch_B', type=str, default=0, help='2nd prediction best_epoch')
parser.add_argument('--epoch_C', type=str, default=0, help='3rd prediction best_epoch')
args = parser.parse_args()

single_dir = '../../datasets/results/Single_model/'                                    # mother directory for single model predictions
ensemble_result_dir = '../../datasets/results/Ensemble_model/' + args.ensemble_type   # directory for Ensemble model predictions
device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu") 

main(args)
