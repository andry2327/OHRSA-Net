import torch
from torch import nn

from torchvision.ops import MultiScaleRoIAlign

from .faster_rcnn import FasterRCNN, TwoMLPHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from GraFormer.network.GraFormer import GraFormer, adj_mx_from_edges
from GraFormer.network.MeshGraFormer import MeshGraFormer

from GraFormer.common.data_utils import create_edges

import sys
sys.path.append('/content/OHRSA-Net/ultralytics')
from ultralytics.models import YOLO

__all__ = [
    "KeypointRCNN", "keypointrcnn_resnet50_fpn"
]


class OHRSA(nn.Module):

    def __init__(self, keypoints2d_extractor_path, num_classes=None,
                 num_kps2d=21, num_kps3d=50, num_verts=2556, photometric=False, hid_size=128,
                 graph_input='coords', num_features=2048, device='cuda', dataset_name='h2o',
                 # additional
                 hands_connectivity_type='',
                 multiframe=False):
        super(OHRSA, self).__init__()

        self.device = device
        self.num_classes = num_classes - 1
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
            self.feature_extractor = TwoMLPHead(256 * 34 * 60, num_features)
            input_size += num_features
            output_size = 3
            if photometric:
                output_size += 3
            self.mesh_graformer = MeshGraFormer(initial_adj=adj.to(device), hid_dim=num_features // 4, coords_dim=(input_size, output_size), 
                            num_kps3d=num_kps3d, num_verts=num_verts, dropout=0.25, 
                            adj_matrix_root='/content/OHRSA-Net/GraFormer/adj_matrix')

    def forward(self, images, targets=None):
        
        # get 2d keypoints and feature maps
        out = self.keypoints2d_extractor(images)
        keypoints2d = out['keypoints']
        feature_maps = out['feature_maps']
        
        batch, kps, dimension = keypoints2d.shape
        graformer_inputs = keypoints2d.view(batch, self.num_classes * kps, dimension)[:, :self.num_kps3d, :2]                
            
        # Estimate 3D pose
        
        # with open(log_time_file_path, 'a') as file:
        #     file.write(f'{datetime.datetime.now()} | START keypoints3d prediction\n')
        keypoint3d = self.keypoint_graformer(graformer_inputs)
        # with open(log_time_file_path, 'a') as file:
        #     file.write(f'{datetime.datetime.now()} | END keypoints3d prediction\n')
        
        # Extract features from feature_maps
        graformer_features = feature_maps
        graformer_features = self.feature_extractor(graformer_features)
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
        