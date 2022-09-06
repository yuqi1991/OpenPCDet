import argparse
import glob,time
from pathlib import Path
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import rospy,ros_numpy
from sensor_msgs.point_cloud2 import PointCloud2,read_points,read_points_list


from visual_utils.ros_visualizer import Visualizer as RosVisualizer

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
        elif self.ext == ".pcd":
            raise NotImplementedError
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
    def __init__(self,args,cfg,logger,demo_dataset):
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        self.model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()
        self.vis = RosVisualizer()
        self.demo_dataset = demo_dataset
        self.logger = logger
        pcd_topic = "/blrobot/drivers/lidar/rslidar_points/top"
        self.subcriber = rospy.Subscriber(pcd_topic, PointCloud2, self.callback)
        self.frame_id = 0


    def remap_result(self,pred_dicts, score_threadhold = 0.4):
        scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        valids = scores > score_threadhold
        labels = pred_dicts[0]['pred_labels'].cpu().numpy()[valids]
        boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()[valids,:]
        scores = scores[valids]
        obj_size = boxes.shape[0]
        dets = np.zeros([obj_size,7])
        reorder = [0,1,2,3,4,5,6]
        dets = boxes[:,reorder]
        other_info = np.concatenate([scores[:,np.newaxis],labels[:,np.newaxis]],axis=1)
        return np.concatenate([dets,other_info],axis=1)

    def get_xyzi_points(self, cloud_array, remove_nans=True, dtype=float):
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

    def callback(self,data):
        start = rospy.Time.now()
        pc_np = self.get_xyzi_points(ros_numpy.point_cloud2.pointcloud2_to_array(data), remove_nans=True).astype(np.float32)
        # pc_np[:,2] -= 0.5
        # pc_np = pc_np[pc_np[:,2] > -1.4]
        pc_np[:,3] = 0.0

        with torch.no_grad():
            data_dict = self.demo_dataset.get_data(self.frame_id,pc_np)
            self.frame_id +=1
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            start_inf = rospy.Time.now()
            pred_dicts, _ = self.model.forward(data_dict)
            end_inf = rospy.Time.now().to_sec() - start_inf.to_sec()
            all_dets = self.remap_result(pred_dicts)
            start_track = rospy.Time.now()
            end_track = rospy.Time.now().to_sec() - start_track.to_sec()
            self.vis.pub_obj(all_dets,data.header.stamp)
            print("whole inference done: {}s, track time: {}s".format(end_inf, end_track))



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
    rospy.init_node("inference")

    infer = Inference(args,cfg,logger,demo_dataset)

    while not rospy.is_shutdown():
        rospy.sleep(0.05)


if __name__ == '__main__':
    main()
