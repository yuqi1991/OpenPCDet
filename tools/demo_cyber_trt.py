import argparse
import glob,time
from pathlib import Path
import cyber_lidar_frame

import struct
import tensorrt as trt
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import pycuda.driver as cuda
import pycuda.autoinit

import sys
sys.path.append("/home/nio/Workspace/AB3DMOT/AB3DMOT_libs")

from google.protobuf import text_format
from python_wrapper import cyber,cyber_time
from point_cloud_pb2 import LidarRaw
from perception_object_pb2 import PerceptionObjects,PerceptionObject

# from sensor_msgs.point_cloud2 import PointCloud2,read_points,read_points_list
# from nio_msgs.msg import PerceptionObject,PerceptionObjects
# from visualization_msgs.msg import Marker, MarkerArray
# import ros_numpy



from AB3DMOT_libs.model import AB3DMOT
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing

class_colors = {
    1:(1.0, 0., 0.),
    2:(1., 0., 1.),
    3:(0., 1., 0.),
    4:(0., 0., 1.),
    5:(0., 1., 1.),
    6:(0., 1., 1.),
}

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


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


class Visualizer(object):
    def __init__(self):
        self.publisher_objs = cyber.Writer("/perception/objects",PerceptionObjects,8)
        # self.publisher_marks = rospy.Publisher('/objects', MarkerArray, queue_size=10)

    def pub(self,pred_dict,stamp):
        obj_size = pred_dict.shape[0]
        is_tracked = pred_dict.shape[1] == 10
        # markerArray = MarkerArray()
        percep_objs = PerceptionObjects()
        percep_objs.header.frame_id = "lidar"
        percep_objs.header.stamp = stamp

        for i in range(obj_size):
            # marker = Marker()
            per_obj = PerceptionObject()
            # marker.header.frame_id = "lidar"
            # marker.type = marker.CUBE
            # marker.action = marker.ADD
            if is_tracked:
                # marker.id = int(pred_dict[i][7])
                per_obj.id = int(pred_dict[i][7])
                label = int(pred_dict[i][9])
            else:
                # marker.id = i
                per_obj.id = i
                label = int(pred_dict[i][8])
            # marker.color.a = 1.0

            # marker.color.r = class_colors[label][0]
            # marker.color.g = class_colors[label][1]
            # marker.color.b = class_colors[label][2]
            # marker.scale.x = pred_dict[i][4]
            # marker.scale.y = pred_dict[i][5]
            # marker.scale.z = pred_dict[i][6]
            # quaternion = euler_to_quaternion(pred_dict[i][3].item(),0, 0)
            # marker.pose.orientation.x = quaternion[0]
            # marker.pose.orientation.y = quaternion[1]
            # marker.pose.orientation.z = quaternion[2]
            # marker.pose.orientation.w = quaternion[3]
            # marker.pose.position.x = pred_dict[i][0]
            # marker.pose.position.y = pred_dict[i][1]
            # marker.pose.position.z = pred_dict[i][2]
            #
            # marker.header.stamp=stamp
            # marker.lifetime = rospy.Duration(0,100000000)
            # markerArray.markers.append(marker)

            per_obj.bounding_box.x = pred_dict[i][4]
            per_obj.bounding_box.y = pred_dict[i][5]
            per_obj.bounding_box.z = pred_dict[i][6]
            per_obj.position.x = pred_dict[i][0]
            per_obj.position.y = pred_dict[i][1]
            per_obj.position.z = pred_dict[i][2]
            per_obj.velocity.x = 0.0
            per_obj.velocity.y = 0.0
            per_obj.velocity.z = 0.0
            per_obj.heading = pred_dict[i][3]
            per_obj.type = label
            percep_objs.objects.append(per_obj)

        # self.publisher_marks.publish(markerArray)
        self.publisher_objs.publish(percep_objs)

def allocate_buffers(engine, pfe_inputs,is_explicit_batch=False, input_shape=None):
    inputs = []
    outputs = []
    bindings = []

    for index, binding in enumerate(engine):
        dims = engine.get_binding_shape(binding)
        for dim_index,dim in enumerate(dims):
            if dim == -1:
                assert(input_shape[index][dim_index] is not None)
                dims[dim_index] = input_shape[index][dim_index]
        if engine.binding_is_input(binding):        #Determine whether a binding is an input binding.
            if type(pfe_inputs[index]) == np.ndarray:
                device_mem = torch.from_numpy(pfe_inputs[index]).cuda()
            elif type(pfe_inputs[index]) == torch.Tensor:
                device_mem = pfe_inputs[index]
            else:
                device_mem = torch.FloatTensor(0).cuda()
            inputs.append(device_mem)
        else:
            size = trt.volume(dims) * engine.max_batch_size #The maximum batch size which can be used for inference.
            device_mem = torch.Tensor(size).to(torch.float32).cuda()
            outputs.append(device_mem)
        bindings.append(int(device_mem.data_ptr()))
    return inputs, outputs, bindings

class Inference:
    def __init__(self,args,cfg,logger,demo_dataset,cyber_node):
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        self.model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()
        self.vis = Visualizer()
        self.tracker = AB3DMOT()
        self.demo_dataset = demo_dataset
        self.logger = logger
        pcd_topic = "/sensing/lidar/combined"
        self.subcriber = cyber_node.create_reader(pcd_topic, LidarRaw, self.callback)
        self.frame_id = 0
        self.input_queue = list()

        # self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        # trt.init_libnvinfer_plugins(None, '')
        # with open("pfe.engine", "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
        #     self.trt_pfe_engine=runtime.deserialize_cuda_engine(f.read())
        #     self.pfe_context = self.trt_pfe_engine.create_execution_context()
        #
        # with open("pp_anchor.engine", "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
        #     self.trt_pp_engine=runtime.deserialize_cuda_engine(f.read())
        #     self.pp_context = self.trt_pp_engine.create_execution_context()

        # pfe_model = onnx.load("pfe.onnx")
        # self.pfe_engine = backend.prepare(pfe_model, device="CUDA:0",max_workspace_size=4<<30)
        #
        # pp_model = onnx.load("pp_anchor_trim.onnx")
        # self.engine_rpn = backend.prepare(pp_model, device="CUDA:0", max_batch_size=1)


    def remap_result(self,pred_dicts, score_threadhold = 0.4):
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

    def infer(self,pfe_inputs):
        pfe_start_time = time.time()

        shapes = [data.shape for data in pfe_inputs]
        voxel_num = pfe_inputs[0].shape[0]
        shapes.append((voxel_num,4))
        segment_inputs, segment_outputs, segment_bindings = allocate_buffers(self.trt_pfe_engine, pfe_inputs,True, shapes)
        pfe_stream = cuda.Stream()

        self.pfe_context.set_optimization_profile_async(0,pfe_stream.handle)
        self.pfe_context.set_binding_shape(0, shapes[0])
        self.pfe_context.set_binding_shape(1, shapes[1])
        self.pfe_context.set_binding_shape(2, shapes[2])

        self.pfe_context.execute_async(bindings=segment_bindings, stream_handle=pfe_stream.handle)
        pfe_stream.synchronize()
        pillar_features = segment_outputs[0].view([voxel_num,64])

        batch_spatial_features = self.model.scatter.export_forward(pillar_features,segment_inputs[2])

        shapes = [batch_spatial_features.shape]
        segment_inputs, segment_outputs, segment_bindings = allocate_buffers(self.trt_pp_engine, batch_spatial_features,True, shapes)
        pp_stream = cuda.Stream()
        self.pp_context.set_optimization_profile_async(0,pp_stream.handle)
        self.pp_context.set_binding_shape(0, shapes[0])
        self.pp_context.execute_async(bindings=segment_bindings, stream_handle=pp_stream.handle)
        batch_cls_preds = segment_outputs[0].view([-1,6])
        batch_box_preds = segment_outputs[1].view([-1,7])
        [final_boxes,final_scores,final_labels] = self.model.post_processing_export(batch_cls_preds,batch_box_preds)
        pfe_end_time = time.time()
        print('inference time is : ', (pfe_end_time - pfe_start_time))


    def callback(self,data):
        start = cyber_time.Time.now()
        pc_np = np.array(cyber_lidar_frame.point_cloud_to_array(data.raw_data))
        pc_np = np.insert(pc_np,3,values= np.zeros((1,pc_np.shape[0])),axis=1)
        # pc_np = self.get_xyzi_points(ros_numpy.point_cloud2.pointcloud2_to_array(data), remove_nans=True).astype(np.float32)
        # pc_np = np.array(0)
        pc_np[:,2] -= 0.5
        pc_np = pc_np[pc_np[:,2] > -1.4]
        # pc_np[:,3] = pc_np[:,3]/255.0

        with torch.no_grad():
            data_dict = self.demo_dataset.get_data(self.frame_id,pc_np)
            self.frame_id +=1
            data_dict = self.demo_dataset.collate_batch([data_dict])
            # load_data_to_gpu(data_dict)
            voxel_features = data_dict['voxels'].astype(np.float32)
            voxel_num_points = data_dict['voxel_num_points'].astype(np.float32)
            coords = data_dict['voxel_coords'].astype(np.float32)
            pfe_inputs = [voxel_features,voxel_num_points,coords]
            # self.infer(pfe_inputs)
        self.input_queue.append(pfe_inputs)

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

    infer = Inference(args,cfg,logger,demo_dataset,perception_node)

    while not cyber.is_shutdown():
        if len(infer.input_queue) > 0:
            infer.infer(infer.input_queue.pop(0))
        cyber_time.Duration(0.05).sleep()


if __name__ == '__main__':
    main()
