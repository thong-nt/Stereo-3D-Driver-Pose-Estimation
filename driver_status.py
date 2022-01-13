import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader 
from modules.pose import Pose

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
        self.l2 =  nn.Linear(64,32)
        self.l3 =  nn.Linear(32,num_classes)
        self.lout =  nn.Linear(num_classes,num_classes)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.elu(self.l2(out))
        out = self.elu(self.l3(out))
        out = self.dropout(out)
        out = self.lout(out)
        return out

def get_package(current_poses):
    features = {"nose_x",	"nose_y",	"neck_x",	"neck_y",	"r_sho_x",	"r_sho_y", "l_sho_x",	"l_sho_y",	"r_eye_x",	"r_eye_y",	"l_eye_x",	"l_eye_y",	"r_ear_x",	"r_ear_y",	"l_ear_x",	"l_ear_y"}
    df = pd.DataFrame(columns=features)
    idx = 0
    for pose in current_poses:
        df=df.fillna(0)
        pose.get_pose_info(None,df,idx)
        idx = idx + 1
    
    pack = np.array(df)
    scaler = StandardScaler()
    pack = scaler.fit_transform(pack)
    input_pack = Input_pack(torch.FloatTensor(pack))
    input_loader = DataLoader(dataset=input_pack, batch_size=1,shuffle=False)
    
    dataiter = iter(input_loader)
    feature = dataiter.next()
    return feature

def get_pos(current_poses, predict_res, img):
    orig_img = img
    id = 0
    _, predicted = torch.max(predict_res, 1)
    state = ["Safe","Left","Right","Up","Down"]
    for pose in current_poses:
        cv2.putText(img, 'Pos: {}'.format(state[predicted[id]]), (20, 20),
                     cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        id = id + 1
    return img
