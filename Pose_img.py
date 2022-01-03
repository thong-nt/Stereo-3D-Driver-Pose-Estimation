import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

from pose3dmodules import *

def run_3dpose(net):
    cpu = False

    img = cv2.imread('img_34.jpg')

    orig_img = img
    Pose.draw(img)
    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    cv2.imshow('image',img)
    key = cv2.waitKey(1)

if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)

    run_3dpose(net)

