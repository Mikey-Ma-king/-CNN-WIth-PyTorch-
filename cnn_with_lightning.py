import torch
import torch.utils.data as u_data

import torch.optim as op
import torch.nn as nn
import torchvision

import lightning as L

classes = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
lea_rate = 0.005

atransform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])])

class LitCNN(L.LightningModule):
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
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        image, label = batch
        output = self(image)
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(output,label)

        return loss
    
    def test_step(self, batch,batch_idx):

        image, label = batch
        output = self(image)
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(output,label)

        return loss
    
    def train_dataloader(self):
        train_dataset = torchvision.datasets.CIFAR10(root = "./cifer_data",train = True,
                                             transform= atransform,download = True)

        train_loader = u_data.DataLoader(train_dataset,batch_size=8,shuffle = True)

        return train_loader
    
    def test_dataloader(self):
       test_dataset = torchvision.datasets.CIFAR10(root = "./cifer_data",train= False,
                                            transform= atransform,download = True)

       test_loader = u_data.DataLoader(test_dataset,batch_size=1,shuffle=False)

       return test_loader
    
    def configure_optimizers(self):
        pg = [p for p in self.parameters() if p.requires_grad]
        return op.Adam(pg,lea_rate)

if __name__ =="__main__" :
    model = LitCNN()
    trainer = L.Trainer()
    trainer.fit(model)
    










