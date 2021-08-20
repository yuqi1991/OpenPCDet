from .detector3d_template import Detector3DTemplate

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.vfe = self.module_list[0]
        self.scatter = self.module_list[1]
        self.backbone = self.module_list[2]
        self.anchor_head = self.module_list[3]


    def forward(self, input):

        ##### Normal usage #####
        for cur_module in self.module_list:
            input = cur_module(input)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(input)
            return pred_dicts, recall_dicts

        ##### ONNX export #####
        # voxel_features = input[0]
        # voxel_num_points = input[1]
        # coords = input[2]
        # pillar_features = self.vfe.export_forward(voxel_features,voxel_num_points,coords)
        # return pillar_features
        # spatial_features = self.scatter.export_forward(pillar_features,coords)
        # # spatial_features = spatial_features[0]
        # spatial_feat_2d = self.backbone.export_forward(spatial_features)
        # [batch_cls_preds,batch_box_preds] = self.anchor_head.export_onnx(spatial_feat_2d)
        # return [batch_cls_preds,batch_box_preds]


    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
