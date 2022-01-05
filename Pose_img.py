import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

from posedataset import *

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


def run_demo(net, img_dir, height_size, cpu, track, smooth, df, id):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1

    img = cv2.imread(img_dir,cv2.IMREAD_COLOR)
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
    for pose in current_poses:
        pose.get_pose_info(img, df, id)
        #img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

        #for pose in current_poses:
        #    cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
        #                  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        #    if track:
        #        cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
        #                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        #cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)

if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)

    csv_path = '/media/nvidia/USB/driver_imgs_list.csv'
    path_to_ds = "/media/nvidia/USB/train/"

    df = Access_data(csv_path)
    id = 0

    while True: 
        frame_provider = Get_img(path_to_ds,df,id)
        run_demo(net, frame_provider, 256, False, 1, 1,df,id)
        id = id + 1
        print(id)
        if id == 22424:
            print("Done")
            Save_ds(df,'/media/nvidia/USB/driver_imgs_list_updae.csv')
        key = cv2.waitKey(0)
        if key == ord('n'):
            id = id + 1
            print(df)
        if key == ord('q'):
            break
