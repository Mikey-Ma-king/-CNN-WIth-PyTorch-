import sys

import torch
import torch.utils.data as u_data
import torch.nn as nn
import torch.optim as op
import numpy as np

import torchvision

from tqdm import tqdm

classes = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
classes_pred = np.zeros(10)
classes_num = np.zeros(10)

class CNN(nn.Module):
    def __init__(self):
       super().__init__()
       #next_ima_size = (ima_size-filter_size+2*padding)/stride+1
       self.layer1 = nn.Conv2d(3,9,3,1,1)#(32-3+2)/1+1=32
       self.pool = nn.MaxPool2d(4,4)#n/4(twice)
       self.layer2 = nn.Conv2d(9,18,5,1,2)#(8-5+4)/1+1=8
       self.fc1 = nn.Linear(18*2*2,120)#final size
       self.fc2 = nn.Linear(120,80)
       self.fc3 = nn.Linear(80,10)

    def forward(self,x):
        x = self.pool(torch.relu((self.layer1(x))))
        x = self.pool(torch.relu((self.layer2(x))))
        x = x.view(-1,18*2*2)
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x) 

        return x 
    
#from image to tensor    
atransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])])

device = torch.device("cpu")

train_dataset = torchvision.datasets.CIFAR10(root = "./cifer_data",train = True,
                                             transform= atransform,download = True)

train_loader = u_data.DataLoader(train_dataset,batch_size=8,shuffle = True)

test_dataset = torchvision.datasets.CIFAR10(root = "./cifer_data",train= False,
                                            transform= atransform,download = True)

test_loader = u_data.DataLoader(test_dataset,batch_size=4,shuffle=False)

def infer(model,dataset,batch_size,device):
    model.eval()
    acc_num = 0
    classes_pred = np.zeros(10)
    classes_num = np.zeros(10)

    with torch.no_grad():
        for data in dataset:
            datas,label = data
            outputs = model(datas.to(device))
            predictions = torch.argmax(outputs,dim = 1)
            acc_num += torch.eq(predictions,label.to(device)).sum().item()

            for i in range(batch_size):
                if label[i]==predictions[i]:
                    classes_pred[label[i]] +=1
                classes_num[label[i]]+=1

    acc = acc_num/len(dataset)
    return acc

def main(lea_rate = 0.005,turns = 20):
    model = CNN().to(device)
    loss_f = nn.CrossEntropyLoss()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = op.Adam(pg,lea_rate)

    for turn in range(turns) :
        model.train()

        train_bar = tqdm(train_loader, file = sys.stdout, ncols = 100)#显示训练过程，可视化

        for datas in train_bar:
            data,label = datas

            optimizer.zero_grad()
            outputs = model(data.to(device))

            loss = loss_f(outputs,label.to(device))
            loss.backward()
            optimizer.step()

            train_bar.desc = "train turns {}/{} loss {:.4f}".format(turn+1,turns,loss)

        test_acc = infer(model,test_loader,4,device)
        print("test turns {}/{}  test_acc {}".format(turn+1,turns,test_acc))
        test_acc = 0
        
    print("Finish Training")

    for i in range(10):
        print (f"{classes[i]}'s accuracy is {100*classes_pred[i]/classes_num[i]}%.")

if __name__ =="__main__" :
    main()   
    










