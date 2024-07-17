"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
import os

import torch
import torch.nn.functional as F
from typing import List
### DEBUG time
from utils.utils_shared import log_time_file_path
import datetime

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, multiframe):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False
        self.multiframe = multiframe

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))
        if self.training:
            if self.multiframe and 'prev_frames' in images:
                prev_frames = images['prev_frames']
                images = images['inputs']
                images_mf = [[img] + prev_frame for img, prev_frame in zip(images, prev_frames)]
                
                # adapt targets
                targets_mf = []
                for i in range(len(targets)):
                    target = targets[i]
                    new_target = [target for _ in range(len(images_mf[i]))]
                    targets_mf.append(new_target)
                
                for i, (images_sample, targets_sample) in enumerate(zip(images_mf, targets_mf)):
                    images_sample, targets_sample = self.transform(images_sample, targets_sample)
                    images_mf[i], targets_mf[i] = images_sample, targets_sample
            else:
                images = images['inputs']
        
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        
        original_images = [img.permute(1, 2, 0) for img in images]
            
        images, targets = self.transform(images, targets)
            
        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None and not self.multiframe:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))
        if self.multiframe:
            features_list = []
            for images_sample in images_mf:
                features = self.backbone(images_sample.tensors)
                for key in features:
                    features[key] = torch.mean(features[key], dim=0, keepdim=True)
                features_list.append(features)
            features = OrderedDict()
            for key in features_list[0].keys():
                tensors = [feature_dict[key] for feature_dict in features_list]
                # Add padding to tensors if needed, check if all tensors have the same shape first
                if not all(tensor.shape == tensors[0].shape for tensor in tensors): 
                    # Pad the tensors to the same shape if they differ
                    tensors = pad_tensors_to_same_shape(tensors)
                concatenated_tensor = torch.cat(tensors, dim=0)
                features[key] = concatenated_tensor
        else:
            # with open(log_time_file_path, 'a') as file:
            #     file.write(f'{datetime.datetime.now()} | START features extraction\n')
            features = self.backbone(images.tensors) # extract image features
            # with open(log_time_file_path, 'a') as file:
            #     file.write(f'{datetime.datetime.now()} | END features extraction\n')
    
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        # with open(log_time_file_path, 'a') as file:
        #     file.write(f'{datetime.datetime.now()} | START rpn proposals\n')
        proposals, proposal_losses = self.rpn(images, features, targets)
        # with open(log_time_file_path, 'a') as file:
        #     file.write(f'{datetime.datetime.now()} | END rpn proposals\n')
        # with open(log_time_file_path, 'a') as file:
        #     file.write(f'{datetime.datetime.now()} | START roi_heads\n')
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, original_images, targets)
        # with open(log_time_file_path, 'a') as file:
        #     file.write(f'{datetime.datetime.now()} | END roi_heads\n')
        # with open(log_time_file_path, 'a') as file:
        #     file.write(f'{datetime.datetime.now()} | START self.transform.postprocess\n')
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        # with open(log_time_file_path, 'a') as file:
        #     file.write(f'{datetime.datetime.now()} | START self.transform.postprocess\n')

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
        return losses, detections

def pad_tensors_to_same_shape(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Pad a list of tensors to the same shape if they have different shapes.

    Args:
        tensors (List[torch.Tensor]): List of tensors to be padded.

    Returns:
        List[torch.Tensor]: List of padded tensors with the same shape.
    """
    # Determine the maximum shape for each dimension
    max_shape = list(tensors[0].shape)
    for tensor in tensors[1:]:
        for dim in range(len(max_shape)):
            max_shape[dim] = max(max_shape[dim], tensor.shape[dim])

    # Pad each tensor to match the maximum shape
    padded_tensors = []
    for tensor in tensors:
        pad_shape = [(0, max_shape[dim] - tensor.shape[dim]) for dim in range(len(max_shape))]
        pad_shape = [item for sublist in pad_shape for item in sublist]  # Flatten the list of tuples
        padded_tensor = F.pad(tensor, pad_shape, "constant", 0)
        padded_tensors.append(padded_tensor)
    
    return padded_tensors