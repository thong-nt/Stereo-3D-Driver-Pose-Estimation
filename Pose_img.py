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

    net = net.eval()
    net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    imgloc = "img_34.jpg"
    Rimg = cv2.imread(imgloc, cv2.IMREAD_COLOR)
    Limg = cv2.imread(imgloc, cv2.IMREAD_COLOR)

    pose_extraction_scheduler = PoseScheduler()
    pose_infer_scheduler = PoseInferScheduler(net, stride, upsample_ratio, cpu)
    pose_infer_scheduler.start_infer()

    pose3d = None
    Rimg = None
    Limg = None
    newpose, r_current_poses, l_current_poses, Rimg_synced, Limg_synced = False, None, None, None, None
   
    while True:
        pose_infer_scheduler.set_images(Rimg, Limg)

        Lnewpose, r_pose_data, Rimg_sync = pose_infer_scheduler.get_left_pose()
        Rnewpose, l_pose_data, Limg_sync = pose_infer_scheduler.get_right_pose()

        Rheatmaps, Rpafs, Rscale, Rpad = r_pose_data
        Lheatmaps, Lpafs, Lscale, Lpad = l_pose_data

        if pose_extraction_scheduler.is_done():
            pose_extraction_scheduler.schedule_new_stereo_extract("left", Rimg_sync, r_previous_poses, Rheatmaps, Rpafs, Rscale, Rpad, num_keypoints, stride, upsample_ratio)
            pose_extraction_scheduler.schedule_new_stereo_extract("right", Limg_sync, l_previous_poses, Lheatmaps, Lpafs, Lscale, Lpad, num_keypoints, stride, upsample_ratio)

            newpose, r_current_poses, l_current_poses, Rimg_synced, Limg_synced = pose_extraction_scheduler.execute_schedule()

        Rimg = draw_pose(r_current_poses, Rimg_synced)
        Limg = draw_pose(l_current_poses, Limg_synced)

        cv2.imshow('img', np.hstack([Rimg, Limg]))
        key = cv2.waitKey(1)
        if key == ord('q'):
            pose_infer_scheduler.stop_infer()
            return


if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)

    run_3dpose(net)

