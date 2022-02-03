import argparse
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from jsk_recognition_msgs.msg import BoundingBoxArray,BoundingBox
import sys

sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages/rospy")
sys.path.append("/opt/ros/melodic/lib/python2.7/dist-packages/tf")

import rospy
from visual_utils.ros_visualizer import Visualizer as RosVisualizer

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

class Visualizer:
    def __init__(self,args,cfg,logger,demo_dataset):
        self.vis = RosVisualizer()
        self.demo_dataset = demo_dataset
        self.logger = logger
        self.frame_cnt = 0


    def get_xyzi_points(self, cloud_array, remove_nans=True, dtype=np.float):
        '''Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
        '''
        # remove crap points
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) & np.isfinite(cloud_array['intensity'])
            cloud_array = cloud_array[mask]

        # pull out x, y, and z values
        points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']
        points[...,3] = cloud_array['intensity']
        return points

    def process(self,data):
        timestamp = rospy.Time.now()
        pc_np = data['points'][:,:3]
        self.vis.pub_pc(pc_np,timestamp.to_nsec())
        self.frame_cnt +=1
        gt_boxes = data['gt_boxes']

        obj_size = gt_boxes.shape[0]
        if obj_size == 0:
            return

        bbox_array = BoundingBoxArray()
        bbox_array.header.frame_id = "base_link"
        bbox_array.header.stamp = timestamp

        for i in range(obj_size):
            bbox = BoundingBox()
            bbox.header.frame_id = "base_link"
            bbox.header.stamp = timestamp
            quaternion = euler_to_quaternion(gt_boxes[i][6].item(),0, 0)
            bbox.pose.orientation.x = quaternion[0]
            bbox.pose.orientation.y = quaternion[1]
            bbox.pose.orientation.z = quaternion[2]
            bbox.pose.orientation.w = quaternion[3]
            bbox.pose.position.x = gt_boxes[i][0]
            bbox.pose.position.y = gt_boxes[i][1]
            bbox.pose.position.z = gt_boxes[i][2]
            bbox.dimensions.x = gt_boxes[i][3]
            bbox.dimensions.y = gt_boxes[i][4]
            bbox.dimensions.z = gt_boxes[i][5]
            bbox.label = int(gt_boxes[i][7])
            bbox_array.boxes.append(bbox)
        self.vis.publisher_bboxs.publish(bbox_array)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/kitti_dataset.yaml',
                        help='specify the config for demo')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')


    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=2,
        logger=logger,
        training=True,
        total_epochs=1
    )
    logger.info(f'Total number of samples: \t{len(train_set)}')
    rospy.init_node("inference")

    vis = Visualizer(args, cfg, logger, train_set)

    with torch.no_grad():
        for idx, data_dict in enumerate(train_set):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            vis.process(data_dict)
            next = input('Press Enter')
    logger.info('Done.')

if __name__ == '__main__':
    main()
