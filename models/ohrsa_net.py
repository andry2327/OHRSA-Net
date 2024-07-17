import torch
from torch import nn

from torchvision.ops import MultiScaleRoIAlign

from .faster_rcnn import FasterRCNN, TwoMLPHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from GraFormer.network.GraFormer import GraFormer, adj_mx_from_edges
from GraFormer.network.MeshGraFormer import MeshGraFormer

from GraFormer.common.data_utils import create_edges

from ultralytics import YOLO

__all__ = [
    "KeypointRCNN", "keypointrcnn_resnet50_fpn"
]


class OHRSA(nn.Module):

    def __init__(self, keypoints2d_extractor_path, num_classes=None,
                 # transform parameters
                 min_size=None, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # keypoint parameters
                 keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None, 
                 num_kps2d=21, num_kps3d=50, num_verts=2556, photometric=False, hid_size=128,
                 graph_input='heatmaps', num_features=2048, device='cuda', dataset_name='h2o',
                 # additional
                 hands_connectivity_type='',
                 multiframe=False):

        self.device = device
        self.num_classes = num_classes
        self.multiframe = multiframe
        
        # Keypoints 2D extractor
        self.keypoints2d_extractor = YOLO(keypoints2d_extractor_path) 

        # GraFormer
        if graph_input == 'heatmaps':          
            input_size = 3136
        else:
            input_size = 2
        
        edges = create_edges(num_nodes=num_kps3d, connectivity_type=hands_connectivity_type)
        adj = adj_mx_from_edges(num_pts=num_kps3d, edges=edges, sparse=False)            
        self.keypoint_graformer = GraFormer(adj=adj.to(device), hid_dim=hid_size, coords_dim=(input_size, 3), 
                                        n_pts=num_kps3d, num_layers=5, n_head=4, dropout=0.25)
        
        # Coarse-to-fine GraFormer
        mesh_graformer = None
        if num_verts > 0:
            feature_extractor = TwoMLPHead(256 * 14 * 14, num_features)
            input_size += num_features
            output_size = 3
            if photometric:
                output_size += 3
            self.mesh_graformer = MeshGraFormer(initial_adj=adj.to(device), hid_dim=num_features // 4, coords_dim=(input_size, output_size), 
                            num_kps3d=num_kps3d, num_verts=num_verts, dropout=0.25, 
                            adj_matrix_root='/content/THOR-Net/GraFormer/adj_matrix')

    def forward(self, images, targets=None):
        
        out = self.keypoints2d_extractor(images)
        keypoints2d = out['keypoints']
        
        if self.num_verts > 0 and ((self.num_classes > 2 and filtered_keypoint_proposals is not None) or batch > 0):

            if self.graph_input == 'heatmaps': # TODO
                batch, kps, H, W = graformer_keypoint_logits.shape                   
                graformer_inputs = graformer_keypoint_logits.view(batch, kps, W * H)
            else:
                batch, kps, dimension = keypoints2d.shape
                graformer_inputs = keypoints2d.view(batch, self.num_classes * kps, dimension)[:, :self.num_kps3d, :2]                
            
            # Estimate 3D pose
            # with open(log_time_file_path, 'a') as file:
            #     file.write(f'{datetime.datetime.now()} | START keypoints3d prediction\n')
            keypoint3d = self.keypoint_graformer(graformer_inputs)
            # with open(log_time_file_path, 'a') as file:
            #     file.write(f'{datetime.datetime.now()} | END keypoints3d prediction\n')
            
            # Extract features from RoIs
            graformer_features = self.feature_extractor(graformer_features)
            
            # Every image has 3 RoIs 
            if self.num_classes > 2:
                graformer_features = graformer_features.view(num_images, num_classes, -1).unsqueeze(axis=2).repeat(1, 1, self.num_kps2d, 1)
                graformer_features = graformer_features.view(num_images, num_classes * kps, 2048)[:, :self.num_kps3d]
            else:
                graformer_features = graformer_features.unsqueeze(axis=1).repeat(1, self.num_kps2d, 1)
            
            # Pass features and pose to Coarse-to-fine GraFormer
            mesh_graformer_inputs = torch.cat((graformer_inputs, graformer_features), axis=2)
            # with open(log_time_file_path, 'a') as file:
            #     file.write(f'{datetime.datetime.now()} | START mesh3d prediction\n')
            mesh3d = self.mesh_graformer(mesh_graformer_inputs)
            # with open(log_time_file_path, 'a') as file:
            #     file.write(f'{datetime.datetime.now()} | END mesh3d prediction\n')
            
        results = {
            'keypoint3d': keypoint3d,
            'mesh3d': mesh3d
        }
        
        return results
        