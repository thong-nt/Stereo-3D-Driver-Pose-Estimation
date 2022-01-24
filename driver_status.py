import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader 
from modules.pose import Pose

state = ["Safe","Distracted"]

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

def get_res(current_poses, model, img):
    features = ["nose_x", "nose_y", "neck_x", "neck_y",	"r_sho_x", "r_sho_y", "l_sho_x", "l_sho_y",	"r_eye_x",	"r_eye_y",	"l_eye_x",	"l_eye_y",	"r_ear_x",	"r_ear_y",	"l_ear_x",	"l_ear_y"]
    df = pd.DataFrame(columns=features)
    idx = 0

    for pose in current_poses:
        pose.get_pose_info(None,df,idx)
        idx = idx + 1
    df=df.fillna(0)

    scaler = StandardScaler()
    pack = scaler.fit_transform(df)
    input_pack = Input_pack(torch.FloatTensor(pack))
    input_loader = DataLoader(dataset=input_pack, batch_size=1,shuffle=False)

    with torch.no_grad():
           for feature in input_loader:
             predict = model(feature)
             print(predict)
             _, predicted = torch.max(predict, 1)
             print(predicted)
             cv2.putText(img, 'Pos: {}'.format(state[predicted[0]]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    
    return img
