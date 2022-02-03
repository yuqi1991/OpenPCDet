import copy
import pickle, json

import numpy as np
from skimage import io

from pcdet.datasets.kitti import kitti_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.dataset import DatasetTemplate


class ONCEDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.root_split_path = self.root_path

        self.sample_id_list = self.dataset_cfg.SPLIT_INFO[self.mode]

        self.all_infos = []
        self.include_once_data(self.mode)

    def include_once_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading ONCE dataset')
        all_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                all_infos.extend(infos)

        self.all_infos.extend(all_infos)

        if self.logger is not None:
            self.logger.info('Total samples for ONCE dataset: %d' % (len(all_infos)))

    def set_split(self, split):

        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        # self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.root_split_path = self.root_path

        self.sample_id_list = self.dataset_cfg.SPLIT_INFO['train' if split is 'train' else 'test']

    def get_lidar(self, idx):
        sequence_id = idx[:6]
        frame_id = idx[6:]
        lidar_file = self.root_split_path/ 'data' / sequence_id / 'lidar_roof' / ('%s.bin' % frame_id)
        assert lidar_file.exists()
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        # points = points[points[:,2] > -1.4]
        points[:, 3] = 0.0
        return points

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, frame):
        return frame['annos']

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=12, has_label=True, count_inside_pts=False, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(frame):
            sequence_id = frame['sequence_id']
            frame_id = frame['frame_id']
            print('%s sequence id: %s - frame id %s \n' % (self.split, sequence_id, frame_id))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sequence_id + frame_id}
            info['point_cloud'] = pc_info

            # image_info = {'image_idx': sample_idx, 'image_shape': 0}
            # info['image'] = image_info
            # calib = self.get_calib(sample_idx)

            # P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            # R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            # R0_4x4[3, 3] = 1.
            # R0_4x4[:3, :3] = calib.R0
            # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            # calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            # info['calib'] = calib_info
            # has_label = hasattr()
            if has_label and 'annos' in frame:
                obj_list = self.get_label(frame)
                annotations = {}
                names = obj_list['names']
                boxes_3d = obj_list['boxes_3d']

                annotations['name'] = np.array([name for name in names])
                # annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                # annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                # annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                # annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[box3d[3], box3d[5],
                                                       box3d[4]] for box3d in boxes_3d])  # lhw(camera) format
                annotations['location'] = np.array([[box3d[0], box3d[1],
                                                     box3d[2]] for box3d in boxes_3d])
                annotations['rotation_y'] = np.array([box3d[6] for box3d in boxes_3d])
                # annotations['score'] = np.array([obj.score for obj in obj_list])
                # annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([name for name in names
                                   if name != 'DontCare'])
                num_gt = len(boxes_3d)
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                if num_objects > 0:
                    loc_lidar = annotations['location'][:num_objects]
                    dims = annotations['dimensions'][:num_objects]
                    rots = annotations['rotation_y'][:num_objects]
                    # loc_lidar = calib.rect_to_lidar(loc)
                    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    # loc_lidar[:, 2] += h[:, 0] / 2
                    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar
                else:
                    # annotations['gt_boxes_lidar'] = np.empty([1,7])
                    annotations['gt_boxes_lidar'] = np.ndarray(0)

                info['annos'] = annotations

                # if count_inside_pts:
                #     points = self.get_lidar(sample_idx)
                #     calib = self.get_calib(sample_idx)
                #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
                #
                #     fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                #     pts_fov = points[fov_flag]
                #     corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                #     num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                #
                #     for k in range(num_objects):
                #         flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                #         num_points_in_gt[k] = flag.sum()
                #     annotations['num_points_in_gt'] = num_points_in_gt

                return info
            else:
                return {}

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        infos = []
        import json
        for sample_id in sample_id_list:
            label_json = self.root_split_path / 'data' / sample_id / (sample_id + '.json')
            with open(label_json, 'r') as fjson:
                label_data = json.load(fjson)
            with futures.ThreadPoolExecutor(num_workers) as executor:
                infos.extend(executor.map(process_single_scene, label_data['frames']))
        infos = [info for info in infos if len(info) > 0]
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('dr_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            if num_obj > 0:
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = pred_boxes
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            # pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    # bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(loc)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 # bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.all_infos[0].keys():
            return None, {}

        import pcdet.datasets.dr.eval as dr_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.all_infos]
        ap_result_str, ap_dict = dr_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.all_infos) * self.total_epochs

        return len(self.all_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.all_infos)

        info = copy.deepcopy(self.all_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        # img_shape = info['image']['image_shape']
        # calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            # 'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')

            gt_names = annos['name']
            # print(loc.shape,dims.shape,np.expand_dims(rots,axis=1).shape)
            if gt_names.shape[0]:
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_boxes_lidar = np.concatenate([loc, dims, np.expand_dims(rots, axis=1)], axis=1).astype(np.float32)
            # gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            else:
                gt_boxes_lidar = np.empty([0, 0])

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            # if self.dataset_cfg.FOV_POINTS_ONLY:
            #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
            #     fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            #     points = points[fov_flag]
            input_dict['points'] = points

        # if "images" in get_item_list:
        #     input_dict['images'] = self.get_image(sample_idx)

        # if "depth_maps" in get_item_list:
        #     input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        # if "calib_matricies" in get_item_list:
        #     input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        data_dict = self.prepare_data(data_dict=input_dict)

        # data_dict['image_shape'] = img_shape
        return data_dict


def create_once_infos(dataset_cfg, class_names, data_path, save_path, workers=12):
    dataset = ONCEDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('once_infos_%s.pkl' % train_split)
    val_filename = save_path / ('once_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'once_infos_trainval.pkl'
    test_filename = save_path / 'once_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    dr_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(train_filename, 'wb') as f:
        pickle.dump(dr_infos_train, f)
    print('ONCE train info file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    dr_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(val_filename, 'wb') as f:
        pickle.dump(dr_infos_val, f)
    print('ONCE val info file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(dr_infos_train + dr_infos_val, f)
    print('ONCE trainval info file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # dr_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(dr_infos_test, f)
    # print('DR info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    # if sys.argv.__len__() > 1 and sys.argv[1] == 'create_once_infos':
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    dataset_cfg = EasyDict(yaml.load(open(sys.argv[1])))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    create_once_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist'],
        data_path=ROOT_DIR / 'data' / 'once',
        save_path=ROOT_DIR / 'data' / 'once'
    )
