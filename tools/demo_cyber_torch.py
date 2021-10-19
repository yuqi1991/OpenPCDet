import argparse
import glob,time
from pathlib import Path
import cyber_lidar_frame

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import sys
sys.path.append("/home/nio/Workspace/AB3DMOT/AB3DMOT_libs")

from google.protobuf import text_format
from python_wrapper import cyber,cyber_time
from point_cloud_pb2 import LidarRaw
from visual_utils.cyber_visualizer import Visualizer as CyberVisualizer
from visual_utils.ros_visualizer import Visualizer as RosVisualizer


from AB3DMOT_libs.model import AB3DMOT
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing

import rospy

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def get_data(self,index,points):
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict



class Inference:
    def __init__(self,args,cfg,logger,demo_dataset,cyber_node):
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        self.model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()
        self.vis = CyberVisualizer(cyber_node)
        self.ros_vis = RosVisualizer()
        self.tracker = AB3DMOT()
        self.demo_dataset = demo_dataset
        self.logger = logger
        pcd_topic = "/sensing/lidar/combined"
        self.subcriber = cyber_node.create_reader(pcd_topic, LidarRaw, self.callback)
        self.frame_id = 0
        self.input_queue = list()


    def remap_result(self,pred_dicts, score_threadhold = 0.45):
        scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        valids = scores > score_threadhold
        labels = pred_dicts[0]['pred_labels'].cpu().numpy()[valids]
        boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()[valids,:]
        scores = scores[valids]
        obj_size = boxes.shape[0]
        dets = np.zeros([obj_size,7])
        reorder = [0,1,2,6,3,4,5]
        dets = boxes[:,reorder]
        other_info = np.concatenate([scores[:,np.newaxis],labels[:,np.newaxis]],axis=1)
        return np.concatenate([dets,other_info],axis=1)


    def infer(self,data):
        start_inf = cyber_time.Time.now()
        timestamp = cyber_lidar_frame.get_time_stamp(data.raw_data)
        pc_np = np.array(cyber_lidar_frame.point_cloud_to_array(data.raw_data))
        pc_np = pc_np[pc_np[:,2] !=0.0]
        pc_np[:,2] += 0.4
        pc_np = pc_np[pc_np[:,2] > -1.5]


        # self.ros_vis.pub_pc(pc_np,timestamp)
        pc_np = np.insert(pc_np,3,values= np.zeros((1,pc_np.shape[0])),axis=1)

        with torch.no_grad():
            data_dict = self.demo_dataset.get_data(self.frame_id,pc_np)
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            pred_dicts, _ = self.model.forward(data_dict)
            all_dets = self.remap_result(pred_dicts)

            self.vis.pub(all_dets,timestamp)
            # self.ros_vis.pub_obj(all_dets,timestamp)
            end_inf = cyber_time.Time.now()
            print('inference time is : ', (end_inf - start_inf).to_sec())


    def callback(self,data):
        self.frame_id +=1
        self.input_queue.append(data)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    cyber.init()
    perception_node = cyber.Node("perception")

    rospy.init_node("perception")

    infer = Inference(args,cfg,logger,demo_dataset,perception_node)

    while not cyber.is_shutdown():
        if len(infer.input_queue) > 0:
            infer.infer(infer.input_queue.pop(0))
        cyber_time.Duration(0.001).sleep()


if __name__ == '__main__':
    main()
