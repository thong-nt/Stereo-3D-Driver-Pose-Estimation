import argparse

import cv2
import numpy as np
import torch
from torch import nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader 
from modules.pose import Pose

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

state = ["Safe","Distracted"]
features = ["nose_x", "nose_y", "neck_x", "neck_y",	"r_sho_x", "r_sho_y", "l_sho_x", "l_sho_y",	"r_eye_x",	"r_eye_y",	"l_eye_x",	"l_eye_y",	"r_ear_x",	"r_ear_y",	"l_ear_x",	"l_ear_y"]

y = 160
h = 230
x = 190
w = 330

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

class Input_pack(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
    def __len__ (self):
        return len(self.X_data)

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, img_dir, height_size, cpu, track, smooth, df, id, model):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1

    img = cv2.imread(img_dir,cv2.IMREAD_COLOR)
    img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (360, 270))
    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
    
    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    if track:
        track_poses(previous_poses, current_poses, smooth=smooth)
        previous_poses = current_poses

    df = pd.DataFrame(columns=features)
    idx = 0

    for pose in current_poses:
        pose.get_pose_info(None,df,idx)
        idx = idx + 1
    df=df.fillna(0)
    #print(df)
    scaler = StandardScaler()
    pack = scaler.fit_transform(df)
    input_pack = Input_pack(torch.FloatTensor(pack))
    input_loader = DataLoader(dataset=input_pack, batch_size=1,shuffle=False)

    with torch.no_grad():
        for features_test in input_loader:
            outputs = model(features_test)
            _, predicted = torch.max(outputs, 1)
            if predicted.shape[0] > 0:
                cv2.putText(img, 'Pos: {}'.format(state[predicted[0]]), (20, 20),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

    return img
    #img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

    #for pose in current_poses:
    #    cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
    #                  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
    #    if track:
    #        cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
    #                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    #cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)

def get_pos(current_poses, predict_res, img):
    orig_img = img

    _, predicted = torch.max(predict_res, 1)
    state = ["Safe","Distracted"]
    print(predicted)
    if predicted.shape[0] > 0:
       cv2.putText(img, 'Pos: {}'.format(state[predicted[0]]), (20, 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    return img

def Get_img(path_to_ds,df,indx):
    img_dir = path_to_ds+df['img'][indx]
    #img = cv2.imread(img_dir,cv2.IMREAD_COLOR)
    return img_dir

if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cpu')
    PATH = '/content/drive/MyDrive/Colab Notebooks/model/head_pose_checkpoint3.pth'
    model = NeuralNetwork(16,64,2)
    model = torch.load(PATH)
    model.eval()
    load_state(net, checkpoint)

    csv_path = '/media/nvidia/USB/test.csv'
    path_to_ds = "/media/nvidia/USB/Test/"

    df = Access_data(csv_path)
    print(df)
    id = 0

    while True: 
        frame_provider = Get_img(path_to_ds,df,id)
        img = run_demo(net, frame_provider, 256, False, 1, 10,df,id,model)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(0)
        if key == ord('n'):
            id = id + 1
            print(df)
        if key == ord('q'):
            break
