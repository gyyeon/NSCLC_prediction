 # -*- coding: utf-8 -*- 
from __future__ import division, print_function

import imageio
import os  
import argparse
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support)
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


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
        
# Evaluate
def evaluate_model(test_dl, model):                                                        
    model.eval() 
    recur_prob = np.array([]) 
    predictions, actuals = list(), list()         
    for sample in test_dl: 
        inputs = sample['image']
        targets = sample['target']
        inputs = inputs.to(device) 
        targets = targets.to(device) 
        output = model(inputs)  
        output = output.detach().cpu().numpy()        
        actual = targets.cpu().numpy()
        actual = actual.reshape((len(actual), 1))
        recur_prob = np.append(recur_prob,output) 
        output_r = output.round()         # round to class values

        predictions.append(output_r)  
        actuals.append(actual)      
    auc = roc_auc_score(np.vstack(actuals), np.vstack(recur_prob)) 
    scores = precision_recall_fscore_support(np.vstack(actuals), np.vstack(predictions)) 
    acc = accuracy_score(np.vstack(actuals), np.vstack(predictions)) 
    predictions = sum([list(map(int, i)) for i in predictions], [])  
    actuals = sum([list(map(int, i)) for i in actuals], [])

    return auc, acc, predictions, actuals, recur_prob, scores


# Data augmentation           
data_transforms = {  
    'test': transforms.Compose([   
        transforms.ToPILImage(), 
        transforms.Grayscale(num_output_channels=1),            
        transforms.ToTensor()
    ])
}

class Subset_dataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform 
    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform:
            x['image'] = self.transform(x['image'])
        return x
    def __len__(self):
        return len(self.subset)

class RecurDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        target_frame: path to csv file with annotations
        root_dir: directory with all images
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

def main(args):  
    if (os.path.isdir('./datasets/results/models'+args.model+'/recur_prob/max')==False): os.mkdir('./datasets/results/models'+args.model+'/recur_prob/max') 
    if (os.path.isdir('./datasets/results/models'+args.model+'/predictions/max')==False): os.mkdir('./datasets/results/models'+args.model+'/predictions/max') 
    if (os.path.isdir('./datasets/Tumors/'+args.model+'/test/max')==False): os.mkdir('./datasets/Tumors/'+args.model+'/test/max') 

    recur_dataset = RecurDataset(csv_file='./csv/sorted_GESIEMENS_530.csv',root_dir='./datasets/2D/530_GESIEMENS/cropped/Resized_total/5mm5slice/max/max_img')      
    recur_df = pd.read_csv('./csv/sorted_GESIEMENS_530.csv')

    total_name_list = recur_df.iloc[:,0].tolist()   
    batch_size = 32
    best_acc = 0
    
    fold1_test_acc, fold2_test_acc, fold3_test_acc, fold4_test_acc, fold5_test_acc, mean_test_acc = [],[],[],[],[],[]
    load_idx = []

    for load in range(0,500):    
        load_idx.append('epoch_'+str(load))
        total_accuracy = 0.0

        f1 = open("./datasets/results/models_idx/noval/530/test_idx.txt", 'r')       
        lines = f1.readlines()

        for j,a in enumerate(lines):   # 5-fold   
            test_idx = list(map(int,a.split())) 
            testset = Subset_dataset([recur_dataset[i] for i in test_idx], data_transforms['test'])    

            modelA = Net()               
            modelA = modelA.to(device)   
                                              
            # Dataloader    
            testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size)
            dataloaders = {'test': testloader}    

            modelA.load_state_dict(torch.load( os.path.join('./datasets/results/models'+args.model, 'max/{}fold-epoch-{}.pth'.format(j+1,load))),strict=False)
            auc, acc, predictions, actuals, recur_prob, scores = evaluate_model(dataloaders['test'], modelA)  
            print(str(j+1) + 'fold____Accuracy: %.3f' % acc) 

            if j==0: fold1_test_acc.append(acc)        
            elif j==1: fold2_test_acc.append(acc)
            elif j==2: fold3_test_acc.append(acc)  
            elif j==3: fold4_test_acc.append(acc)
            else: fold5_test_acc.append(acc)
                
            total_accuracy += acc
            mean_accuracy = total_accuracy/5.0

        if mean_accuracy>=best_acc: 
            best_acc = mean_accuracy
            best_epoch = load
        print('5 fold CV Accuracy_______%depoch : %.3f' % (load,mean_accuracy))
        print('Best Accuracy__________%depoch: %.3f' %(best_epoch, best_acc))
        mean_test_acc.append(mean_accuracy)
    
    df = pd.DataFrame(data={"epoch": load_idx, "eval_acc": mean_test_acc})  
    df.to_csv("./datasets/Tumors/"+args.model+"/test/max/epoch_acc.csv", sep=',',index=False)
    best_epoch_list = [i for i in df.index.tolist() if df.loc[i,'eval_acc']==best_acc]
    

    for epoch in best_epoch_list: 
        f3 = open("./datasets/results/models"+args.model+"/max_metrics_"+str(epoch)+"Epoch.txt", 'w')
        precision_score, f1_score, recall_score, auc_score, acc_list = [],[],[],[],[]

        print('====================Epoch '+str(epoch)+'====================')
        for k,a in enumerate(lines):   # 5-fold   
            test_idx = list(map(int,a.split())) 

            test_fn = [total_name_list[i] for i in test_idx]  
            testset = Subset_dataset([recur_dataset[i] for i in test_idx], data_transforms['test'])    

            modelA = Net()           
            modelA = modelA.to(device)   
            
            testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size)

            modelA.load_state_dict(torch.load(os.path.join('./datasets/results/models'+args.model, 'max/{}fold-epoch-{}.pth'.format(k+1,epoch))), strict=False)
            auc, acc, predictions, actuals, recur_prob, scores = evaluate_model(testloader, modelA)  
            precision_score.append(scores[0])  
            recall_score.append(scores[1])
            f1_score.append(scores[2])
            auc_score.append(auc)     
            acc_list.append(acc)     
            f3.write(str(classification_report(actuals, predictions))+'\n') 

            # prob_data = pd.DataFrame(data={"recur_prob": recur_prob}, index = test_fn)  
            pred_data = pd.DataFrame(data={"predictions": predictions}, index = test_fn)       
            pred_data.to_csv('./datasets/results/models'+args.model+'/predictions/max/predictions_'+str(k+1)+'fold_'+str(epoch)+'Epoch.csv') 
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
parser.add_argument('--model', type=str, default='0', help='model number')
parser.add_argument('--cuda', type=str, default='0', help='cuda number')
parser.add_argument('--seed', type=int, default=42, help='set seed')
args=parser.parse_args()

device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu") 

main(args)
