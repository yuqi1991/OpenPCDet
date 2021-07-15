import argparse
import glob
from pathlib import Path

import numpy as np
import torch,time
import math
import pprint, pickle

from onnxruntime.quantization.onnx_model import ONNXModel

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


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
        return (data_dict,points)


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

def get_obj_corners(tensor):
    obj = tensor.cpu().numpy()
    center = [obj[0], obj[1], obj[2]]  # xyz
    size = [obj[3], obj[4], obj[5]]    # lwh
    yaw = obj[6]  # heading
    rot = np.asmatrix([[math.cos(yaw), -math.sin(yaw)], \
                       [math.sin(yaw),  math.cos(yaw)]])
    plain_pts = np.asmatrix([[0.5 * size[0], 0.5*size[1]], \
                             [0.5 * size[0], -0.5*size[1]], \
                             [-0.5 * size[0], -0.5*size[1]], \
                             [-0.5 * size[0], 0.5*size[1]]])
    tran_pts = np.asarray(rot * plain_pts.transpose());
    tran_pts = tran_pts.transpose()
    corners = np.arange(24).astype(np.float32).reshape(8, 3)
    for i in range(8):
        corners[i][0] = center[0] + tran_pts[i%4][0]
        corners[i][1] = center[1] + tran_pts[i%4][1]
        corners[i][2] = center[2] + (float(i >= 4) - 0.5) * size[2];
    return corners



def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        data_dict = demo_dataset[0]
        data_dict = demo_dataset.collate_batch([data_dict[0]])
        load_data_to_gpu(data_dict)
        voxel_features = data_dict['voxels']
        voxel_num_points = data_dict['voxel_num_points']
        coords = data_dict['voxel_coords']
        input_names = ["voxel_features","voxel_num_points","coords"]
        input = [voxel_features,voxel_num_points,coords]

        # ############### EXPORT  PFE #################
        torch.onnx.export(model, input, "pfe.onnx", verbose=True, input_names=input_names,keep_initializers_as_inputs=True,
                          do_constant_folding=True,
                          output_names = ['batch_cls_preds','batch_box_preds'],opset_version=11,
                          dynamic_axes={
                              "voxel_features":{0:"voxel_num"},
                              "voxel_num_points":{0:"voxel_num"},
                              "coords":{0:"voxel_num"},
                              "pfe_feats":{0:"voxel_num"}
                          })


        data_dict = demo_dataset[2]
        data_dict = demo_dataset.collate_batch([data_dict[0]])
        load_data_to_gpu(data_dict)
        voxel_features = data_dict['voxels']
        voxel_num_points = data_dict['voxel_num_points']
        coords = data_dict['voxel_coords']
        input = [voxel_features,voxel_num_points,coords]
        ############## PFE-Layer TensorRT #####
        import onnx
        import onnxruntime
        import onnxsim
        pfe_model = onnx.load("pfe.onnx")
        logger.info('Demo done.')
        pfe_session = onnxruntime.InferenceSession("pfe.onnx")
        pfe_inputs = {pfe_session.get_inputs()[0].name: (voxel_features.data.cpu().numpy()),
                      pfe_session.get_inputs()[1].name: (voxel_num_points.data.cpu().numpy()),
                      pfe_session.get_inputs()[2].name: (coords.data.cpu().numpy())}
        pfe_outs = pfe_session.run(None, pfe_inputs)
        print('-------------------------- PFE ONNX Outputs ----------------------------')
        print(pfe_outs[0])
        print('------------------------- PFE Pytorch Outputs ---------------------------')
        torch_output = model(input)
        print(torch_output[0].cpu().detach().numpy())
        print("PFE mean error:", np.mean(pfe_outs[0] - torch_output.cpu().detach().numpy()))
        print("PFE max error:",  np.max(np.abs((pfe_outs[0] - torch_output.cpu().detach().numpy()))))
        # print('------------------------- PP ANCHOR TensorRT ---------------------------')
        # print("PFE mean error:", np.mean(pfe_outs[0] - torch_output.cpu().detach().numpy()))
        # print("PFE max error:",
        #       np.max(np.abs((pfe_outs[0] - torch_output.cpu().detach().numpy()))))

        ############### EXPORT  NN #################
        # input_names = ["spatial_features"]
        # pkl_file = open("/home/nio/Workspace/OpenPCDet/output/spatial_feats/1.pkl", 'rb')
        # spatial_features = pickle.load(pkl_file)
        # spatial_features = torch.from_numpy(spatial_features).cuda()
        # spatial_features = torch.ones([1,64,496,864],dtype=torch.float32).cuda()
        # input = [spatial_features]
        # torch.onnx.export(model, input, "pp_anchor.onnx", verbose=True, keep_initializers_as_inputs=True,
        # input_names=input_names,output_names = ['batch_cls_preds','batch_box_preds'],opset_version=11)

        ############## PFE-Layer TensorRT #####
        # import onnx
        # import onnxruntime
        # import onnx_tensorrt.backend as backend
        # pkl_file = open("/home/nio/Workspace/OpenPCDet/output/spatial_feats/2.pkl", 'rb')
        # spatial_features = pickle.load(pkl_file)
        # spatial_features = torch.from_numpy(spatial_features).cuda()
        # input = [spatial_features]
        # logger.info('Demo done.')
        # pp_anchor = onnxruntime.InferenceSession("pp_anchor_sim.onnx")
        # pp_anchor_inputs = {pp_anchor.get_inputs()[0].name: (spatial_features.data.cpu().numpy())}
        # pp_anchor_outs = pp_anchor.run(None, pp_anchor_inputs)
        # print('-------------------------- PP ANCHOR ONNX Outputs ----------------------------')
        # print(pp_anchor_outs[0])
        # print(pp_anchor_outs[1])
        # print('------------------------- PP ANCHOR Pytorch Outputs ---------------------------')
        # torch_output = model(input)
        # print(torch_output[0].cpu().detach().numpy())
        # print(torch_output[1].cpu().detach().numpy())
        # print("PP ANCHOR mean error:", np.mean(pp_anchor_outs[0] - torch_output[0].cpu().detach().numpy())," | ",
        #                          np.mean(pp_anchor_outs[1] - torch_output[1].cpu().detach().numpy()) )
        # print("PP ANCHOR max error:",
        #       np.max(np.abs((pp_anchor_outs[0] - torch_output[0].cpu().detach().numpy()))), " | ",
        #       np.max(np.abs((pp_anchor_outs[1] - torch_output[1].cpu().detach().numpy()))))
        # print('------------------------- PP ANCHOR TensorRT ---------------------------')
        # pp_onnx = onnx.load("pp_anchor_trim.onnx")
        # engine = backend.prepare(pp_onnx, device="CUDA:0", max_batch_size=1)
        # rpn_start_time = time.time()
        # pp_output = engine.run(spatial_features.data.cpu().numpy())
        # rpn_end_time = time.time()
        # print(rpn_end_time - rpn_start_time)


if __name__ == '__main__':
    main()
