# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import itertools

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt
from torchvision.models import resnet18, resnet34, resnet101, resnet152


from spectrogram import Spectrogram

import re




class TestDataset(Dataset):
    
    def __init__(self, file_pth: str, train=True, random_state=None):
        super(Dataset, self).__init__()
        
        self.sp = Spectrogram()
        
        self.file_pth = file_pth
        self.train = train
        self.random_state = None
        
        self.data = self.read_data()
        datasets = train_test_split(self.data)
        if train:
            self.train = datasets[0]
        else:
            # actually this is test dataset
            self.train = datasets[1]
    
    def __getitem__(self, index):
        
        # datasets[0][index][0]
        stage_layer, degc_layer = self.feature_layer(self.train[index][2])
        img = self.sp.spec_array(self.train[index][0])
        img = np.concatenate((img, stage_layer), axis=0)
        img = np.concatenate((img, degc_layer), axis=0)
        img = torch.tensor(img, dtype=torch.float32)
        target = self.train[index][1]
        target = torch.tensor(target)
        return img, target
    
    def __len__(self):
        return len(self.train)


    def feature_layer(self, features):

        stage = int(re.findall("\d+", features[0])[0])
        degc = float(features[1])

        stage_layer = np.full((224,224), stage, dtype = np.int8)
        degc_layer = np.full((224,224), degc, dtype = np.float)

        stage_layer = stage_layer[np.newaxis,:,:]
        degc_layer = degc_layer[np.newaxis,:,:]

        return stage_layer, degc_layer
    
    def read_data(self):
        datas = []
        #self.file = "./sample_data.txt"
        
        with open(self.file_pth, "r") as f:
            header = f.readline()
            while 1:
                line = f.readline()
                if not line:
                    break
                tmp = line.strip().split('\t')
                freq = list(map(float, tmp[4:]))
                features = tmp[2:4]
                # freq = list(map(float, tmp[1:]))
                # print(freq[-1])
                label = int(tmp[0])
                
                datas.append([freq,label, features])
                
        return datas

class SampleDataset(Dataset):
    
    def __init__(self, file_pth: str, train=True, random_state=None):
        super(Dataset, self).__init__()
        
        self.sp = Spectrogram()
        
        self.file_pth = file_pth
        self.train = train
        self.random_state = None
        
        self.train = self.read_data()
    
    def __getitem__(self, index):
        
        # datasets[0][index][0]
        stage_layer, degc_layer = self.feature_layer(self.train[index][2])
        img = self.sp.spec_array(self.train[index][0])
        img = np.concatenate((img, stage_layer), axis=0)
        img = np.concatenate((img, degc_layer), axis=0)
        img = torch.tensor(img, dtype=torch.float32)
        target = self.train[index][1]
        target = torch.tensor(target)
        return img, target
    
    def __len__(self):
        return len(self.train)
    
    def feature_layer(self, features):

        stage = int(re.findall("\d+", features[0])[0])
        degc = float(features[1])

        stage_layer = np.full((224,224), stage, dtype = np.int8)
        degc_layer = np.full((224,224), degc, dtype = np.float)

        stage_layer = stage_layer[np.newaxis,:,:]
        degc_layer = degc_layer[np.newaxis,:,:]

        return stage_layer, degc_layer
    
    def read_data(self):
        datas = []
        #self.file = "./sample_data.txt"
        with open(self.file_pth, "r") as f:
            header = f.readline()
            while 1:
                line = f.readline()
                if not line:
                    break
                tmp = line.strip().split('\t')
                freq = list(map(float, tmp[4:]))
                features = tmp[2:4]
                # freq = list(map(float, tmp[1:]))
                # print(freq[-1])
                label = int(tmp[0])
                
                datas.append([freq,label, features])
                
        return datas

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = resnet34(pretrained=True)
        print(self.model.fc.in_features)
        
        self.fc = nn.Linear(512, 2)
        # self.fc0 = nn.Linear(57800, 34)
        # self.fc1 = nn.Linear(34, 10)
        # self.fc2 = nn.Linear(10, 2)
    
    
    
    
    def forward(self,x):
        x = x.view(-1, 3, 224, 224)
        x = F.relu(self.model(x))
        # x.view(-1, self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))
        # torch.Size([2, 50, 34, 34])
        # print(f'what is this?{x.shape}')
        x = F.relu(self.fc(x))

    #    print(x.shape)
    #     x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        # return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
    #    print(f'size: {size}')
        num_features = 1
        for s in size:
            num_features *= s
    #    print(f'num_features: {num_features}')
    #     57800
        return num_features



def main():
    '''
    f_origin = open('../eddie/ai_championship/data/202004/04/202004_FLD165NBMA_vib_spectrum_modi_train_04_split_002.txt', 'r')
    f_ng = open('../ng.txt', 'r')
    f_new = open('../eddie/ai_championship/data/202004/04/202004_FLD165NBMA_vib_spectrum_modi_train_04_split_002_appendng.txt', 'a')
    num = 0
    while True:
        line = f_origin.readline()
        if not line: break
        f_new.write(line)
        num += 1
        print(f'{num}번째 데이터 저장')

    f_origin.close()
    num=0
    while True:
        line = f_ng.readline()
        if not line: break
        f_new.write(line)
        num += 1
        print(f'{num}번째 데이터 저장')
    f_ng.close()
    f_new.close()
    print("데이터 병합 완료")
    '''
    
    # train_dataset = TestDataset('../augmented.txt')
    # test_dataset = TestDataset('../augmented.txt', train=False)
    # train_dataset = TestDataset('../lg_train/202004_FLD165NBMA_vib_spectrum_modi_train.txt')
    # print("train data load")
    # test_dataset = TestDataset('../lg_train/202004_FLD165NBMA_vib_spectrum_modi_train.txt', train=False)
    # print("test data load")

    train_dataset = TestDataset('../eddie/ai_championship/data/202004/04/202004_FLD165NBMA_vib_spectrum_modi_train_04_split_002_appendng.txt')
    print("train data load")
    test_dataset = TestDataset('../eddie/ai_championship/data/202004/04/202004_FLD165NBMA_vib_spectrum_modi_train_04_split_002_appendng.txt', train=False)
    print("test data load")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2,shuffle=True, num_workers=10)
    print("train data loader")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2,shuffle=True, num_workers=10)
    print("train data loader")
    sample_dataset = SampleDataset('./sample_data.txt')
    sampleloader = torch.utils.data.DataLoader(sample_dataset, batch_size=2,shuffle=True, num_workers=4)


    model_ft = resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # 여기서 각 출력 샘플의 크기는 2로 설정합니다.
    # 또는, nn.Linear(num_ftrs, len (class_names1))로 일반화할 수 있습니다.
    # model_ft.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model_ft.conv1 = nn.Conv2d(5, 64, kernel_size=3)
    model_ft.fc = nn.Linear(num_ftrs, 2)
    print(model_ft)

    model_ft = model_ft.to('cuda')

    criterion = nn.CrossEntropyLoss()

    # cnn = CNN().to('cuda')
    # criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.SGD(cnn.parameters(), lr=0.01)
    # criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=0.01)




    model_ft.train()  # 학습을 위함
    for epoch in range(5):
        print(f'epoch {epoch}')
        for index, (data, target) in enumerate(trainloader):
            print(f'{index}')
            data = data.to('cuda')
            target = target.to('cuda')
            optimizer.zero_grad()  # 기울기 초기화
            output = model_ft(data)
    #         print(f"output : {output}, target:{target}")
            loss = criterion(output, target)
            loss.backward()  # 역전파
            optimizer.step()

        if index % 1 == 0:
            print("loss of {} epoch, {} index : {}".format(epoch, index, loss.item()))



    model_ft.eval()  # test case 학습 방지를 위함
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data = data.to('cuda')
            target = target.to('cuda')
            output = model_ft(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(testloader.dataset),
                100. * correct / len(testloader.dataset)))

    # 모델 저장 및 sample data 정확도 확인
    torch.save(model_ft, '../model/model_peter1.pt')

    # model = torch.load('../model.pt')


    model_ft.eval()  # test case 학습 방지를 위함
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in sampleloader:
            data = data.to('cuda')
            target = target.to('cuda')
            print(f"target : {target}")
            output = model_ft(data)
            print(f"output : {output}")
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            print(f"pred : {pred}")
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(sampleloader.dataset),
                100. * correct / len(sampleloader.dataset)))




if __name__ == "__main__":
    main()
