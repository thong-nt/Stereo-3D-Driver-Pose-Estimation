import numpy as np
import pandas as pd
import torch

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

def get_package(current_poses):
    features = {"nose_x",	"nose_y",	"neck_x",	"neck_y",	"r_sho_x",	"r_sho_y", "l_sho_x",	"l_sho_y",	"r_eye_x",	"r_eye_y",	"l_eye_x",	"l_eye_y",	"r_ear_x",	"r_ear_y",	"l_ear_x",	"l_ear_y"}
    df = pd.DataFrame(features)
    idx = 0
    for pose in current_poses:
        pose.get_pose_info(None,df,idx)
        idx = idx + 1

    pack = np.array(df)
    scaler = StandardScaler()
    pack = scaler.fit_transform(pack)
    input_pack = Input_pack(torch.FloatTensor(pack))

    return input_pack

def get_pos(current_poses, predict_res, img):
    orig_img = img
    id = 0
    _, predicted = torch.max(predict_res, 1)
    state = {"Safe","Left","Right","Up","Down"}
    for pose in current_poses:
        cv2.putText(img, 'Pos: {}'.format(state[predicted[id]]), (20, 20),
                     cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        id = id + 1
    return img