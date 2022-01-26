import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader 
from modules.pose import Pose

state = ["Safe","Distracted"]
scaler = StandardScaler()

class Input_pack(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
    def __len__ (self):
        return len(self.X_data)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sze, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 =  nn.Linear(input_size,64)
        #self.l2 =  nn.Linear(64,32)
        self.l3 =  nn.Linear(64,32)
        self.lout =  nn.Linear(32,2)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.relu(self.l1(x))
        #out = self.elu(self.l2(out))
        out = self.elu(self.l3(out))
        out = self.dropout(out)
        out = self.lout(out)
        return out

def get_res(df,idx,current_poses, model, img):
    for pose in current_poses:
        pose.get_pose_info(img,df,idx)
   
    df=df.fillna(0)
    if idx == 0:
       input_pack = Input_pack(torch.FloatTensor(df.values))
    else:
       pack = scaler.fit_transform(df)
       f = np.array(pack[len(pack)-1])
       input_pack = Input_pack(torch.FloatTensor(np.reshape(f,(1,16))))

    input_loader = DataLoader(dataset=input_pack, batch_size=1,shuffle=False)

    for feature in input_loader:
        predict = model(feature)
        _, predicted = torch.max(predict, 1)
        cv2.putText(img, 'Pos: {}'.format(state[predicted[0]]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    
    return img
