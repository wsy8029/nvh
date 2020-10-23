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
from torchvision.models import resnet34, resnet101, resnet152


from spectrogram import Spectrogram




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
        img = self.sp.spec_array(self.train[index][0])
        img = torch.tensor(img, dtype=torch.float32)
        target = self.train[index][1]
        target = torch.tensor(target)
        return img, target
    
    def __len__(self):
        return len(self.train)
    
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
                # freq = list(map(float, tmp[4:]))
                freq = list(map(float, tmp[1:]))
                # print(freq[-1])
                label = int(tmp[0])
                
                datas.append([freq,label])
                
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
    train_dataset = TestDataset('../augmented.txt')
    test_dataset = TestDataset('../augmented.txt', train=False)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2,shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2,shuffle=False, num_workers=2)


    model_ft = resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # 여기서 각 출력 샘플의 크기는 2로 설정합니다.
    # 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to('cuda')

    criterion = nn.CrossEntropyLoss()


    # cnn = CNN().to('cuda')
    # criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.SGD(cnn.parameters(), lr=0.01)
    # criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=0.01)




    model_ft.train()  # 학습을 위함
    for epoch in range(10):
        for index, (data, target) in enumerate(trainloader):
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

if __name__ == "__main__":
    main()
