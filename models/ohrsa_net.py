import torch
from torch import nn

from torchvision.ops import MultiScaleRoIAlign

from .faster_rcnn import FasterRCNN, TwoMLPHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from GraFormer.network.GraFormer import GraFormer, adj_mx_from_edges
from GraFormer.network.MeshGraFormer import MeshGraFormer

from GraFormer.common.data_utils import create_edges

from utils.utils import get_keypoints_bloodiness

# DEBUG time
from utils.utils_shared import log_time_file_path
import datetime

import sys
sys.path.append('/home/aidara/Desktop/Thesis_Andrea/OHRSA-Net_Experiments/OHRSA-Net/ultralytics')
from ultralytics.models import YOLO

__all__ = [
    "KeypointRCNN", "keypointrcnn_resnet50_fpn"
]

ADJ_MATRIX_ROOT = '/home/aidara/Desktop/Thesis_Andrea/OHRSA-Net_Experiments/OHRSA-Net/GraFormer/adj_matrix'

class OHRSA(nn.Module):

    def __init__(self, keypoints2d_extractor_path, num_classes=None,
                 num_kps2d=21, num_kps3d=50, num_verts=2556, photometric=False, hid_size=128,
                 graph_input='coords', num_features=2048, device='cpu', dataset_name='h2o',
                 # additional
                 hands_connectivity_type='',
                 multiframe=False,
                 bloodiness=False):
        super(OHRSA, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.multiframe = multiframe
        self.bloodiness = bloodiness
        self.num_kps2d = num_kps2d
        self.num_kps3d = num_kps3d
        
        # Keypoints 2D extractor
        self.keypoint_predictor = YOLO(keypoints2d_extractor_path).to(self.device) 
        
        # GraFormer
        if graph_input == 'heatmaps':          
            input_size = 3136
        else:
            input_size = 2
            if self.bloodiness:
                input_size = 3
            
        
        edges = create_edges(num_nodes=num_kps3d, connectivity_type=hands_connectivity_type)
        adj = adj_mx_from_edges(num_pts=num_kps3d, edges=edges, sparse=False)            
        self.keypoint_graformer = GraFormer(adj=adj.to(self.device), hid_dim=hid_size, coords_dim=(input_size, 3), 
                                        n_pts=num_kps3d, num_layers=5, n_head=4, dropout=0.25)
        
        # Coarse-to-fine GraFormer
        mesh_graformer = None
        if num_verts > 0:
            self.feature_extractor = TwoMLPHead(256 * 17 * 17, num_features)
            input_size += num_features
            output_size = 3
            if photometric:
                output_size += 3
            self.mesh_graformer = MeshGraFormer(initial_adj=adj.to(self.device), hid_dim=num_features // 4, coords_dim=(input_size, output_size), 
                            num_kps3d=num_kps3d, num_verts=num_verts, dropout=0.25, 
                            adj_matrix_root=ADJ_MATRIX_ROOT)
    
    def forward(self, images, targets=None, fps=None):
        
        # log_buffer = [] # DEBUG time 

        # Get 2D keypoints and feature maps
        with torch.no_grad():
            # log_buffer.append(f'{datetime.datetime.now()} | START keypoints2d prediction\n') # DEBUG time 
            out = self.keypoint_predictor.predict(images, imgsz=images.shape[-2:], max_det=1, verbose=False)
            # log_buffer.append(f'{datetime.datetime.now()} | END keypoints2d prediction\n') # DEBUG time 

        zero_keypoints = torch.zeros((1, self.num_kps2d, 2), device=self.device)
        keypoints2d_list = [o.keypoints.xy if o.keypoints.xy.numel() > 0 else zero_keypoints for o in out]
        
        if self.bloodiness and fps:
            bloodiness = get_keypoints_bloodiness(keypoints2d_list, fps)
            bloodiness = torch.tensor(bloodiness).unsqueeze(-1)
        
        keypoints2d = torch.cat(keypoints2d_list)
        keypoints2d /= torch.tensor([images.shape[-1], images.shape[-2]], device=self.device)

        feature_maps = out[0].feature_maps
        
        batch, kps, dimension = keypoints2d.shape
        graformer_inputs = keypoints2d.view(batch, -1, dimension)[:, :self.num_kps3d, :2]
        if self.bloodiness:
            graformer_inputs = torch.cat((graformer_inputs, bloodiness), dim=2)

        # log_buffer.append(f'{datetime.datetime.now()} | START keypoints3d prediction\n') # DEBUG time 
        keypoint3d = self.keypoint_graformer(graformer_inputs)
        # log_buffer.append(f'{datetime.datetime.now()} | END keypoints3d prediction\n') # DEBUG time 

        graformer_features = self.feature_extractor(feature_maps)
        graformer_features = graformer_features.unsqueeze(1).expand(-1, self.num_kps2d, -1)

        mesh_graformer_inputs = torch.cat((graformer_inputs, graformer_features), dim=2)

        # log_buffer.append(f'{datetime.datetime.now()} | START mesh3d prediction\n') # DEBUG time 
        mesh3d = self.mesh_graformer(mesh_graformer_inputs)
        # log_buffer.append(f'{datetime.datetime.now()} | END mesh3d prediction\n') # DEBUG time 

        # with open(log_time_file_path, 'a') as file:  # DEBUG time 
        #     file.writelines(log_buffer)

        return {
            'keypoint2d': keypoints2d,
            'keypoint3d': keypoint3d,
            'mesh3d': mesh3d
        }