# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
import cv2
import os.path
import io
import torch 
from PIL import Image
from .rcnn_utils import calculate_bounding_box, create_rcnn_data


class Dataset(data.Dataset):
    """# Dataset Class """

    def __init__(self, root='./', load_set='train', transform=None, num_kps3d=21, num_verts=778, hdf5_file=None, other_params={}):

        self.root = root
        self.transform = transform
        self.num_kps3d = num_kps3d
        self.num_verts = num_verts
        self.hdf5 = hdf5_file
        self.IS_MULTIFRAME = other_params['IS_MULTIFRAME']
        self.N_PREVIOUS_FRAMES = other_params['N_PREVIOUS_FRAMES']
        self.STRIDE_PREVIOUS_FRAMES = other_params['STRIDE_PREVIOUS_FRAMES']

        # TODO: add depth transformation
        self.load_set = load_set  # 'train','val','test'
        self.images = np.load(os.path.join(root, 'images-%s.npy' % self.load_set), allow_pickle=True)
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy' % self.load_set), allow_pickle=True)
        self.points2d = self.points2d.astype(np.float64) if self.points2d.dtype == object else self.points2d
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy' % self.load_set), allow_pickle=True)
        self.points3d = self.points3d.astype(np.float64) if self.points3d.dtype == object else self.points3d

        self.mesh2d = np.load(os.path.join(root, 'mesh2d-%s.npy' % self.load_set), allow_pickle=True)
        self.mesh2d = self.mesh2d.astype(np.float64) if self.mesh2d.dtype == object else self.mesh2d
        self.mesh3d = np.load(os.path.join(root, 'mesh3d-%s.npy' % self.load_set), allow_pickle=True)
        self.mesh3d = self.mesh3d.astype(np.float64) if self.mesh3d.dtype == object else self.mesh3d

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """

        image_path = self.images[index]
        palm = self.points3d[index][0]
        point2d = self.points2d[index]
        point3d = self.points3d[index] - palm # Center around palm
                
        # Load image and apply preprocessing if any
        if self.hdf5 is not None:
            data = np.array(self.hdf5[image_path])
            original_image = np.array(Image.open(io.BytesIO(data)))[..., :3]
        else:
            try:
                original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            except:
                image_path = image_path.replace('gdrive', 'drive') # original files save with path 'content/gdrive/ ...'
                # print(f'DEBUG: {image_path}')
                original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                

        inputs = self.transform(original_image)  # [:3]

        if self.load_set != 'test':
            # Loading 2D Mesh for bounding box calculation
            if self.num_kps3d == 21: #i.e. hand
                mesh2d = self.mesh2d[index][:778]
                mesh3d = self.mesh3d[index][:778] - palm
            else: # i.e. object
                mesh2d = self.mesh2d[index]
                mesh3d = self.mesh3d[index] - palm
      
            bb = calculate_bounding_box(mesh2d, increase=True)
            
            if self.num_verts > 0:
                boxes, labels, keypoints, keypoints3d = create_rcnn_data(bb, point2d, point3d, num_keypoints=self.num_kps3d)
                mesh3d = torch.Tensor(mesh3d[:self.num_verts][np.newaxis, ...]).float()
            else:
                boxes, labels, keypoints, keypoints3d = create_rcnn_data(bb, point2d, point3d, num_keypoints=self.num_keypoints)
                mesh3d = torch.tensor([])
        else:
            bb, mesh2d = np.array([]), np.array([])
            boxes, labels, keypoints, keypoints3d, mesh3d = torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        if self.IS_MULTIFRAME:
            frame, extension = tuple(image_path.split(os.sep)[-1].split('.'))
            prev_frames = []
            for i in range(1, self.N_PREVIOUS_FRAMES+1):
                frame_prev = int(frame) - i*self.STRIDE_PREVIOUS_FRAMES
                frame_prev_path = image_path.replace(f'{frame}.{extension}', f'{frame_prev}'.zfill(5)+'.'+f'{extension}')
                if frame_prev_path in self.images:
                    # Load image and apply preprocessing if any
                    if self.hdf5 is not None:
                        data = np.array(self.hdf5[frame_prev_path])
                        prev_original_image = np.array(Image.open(io.BytesIO(data)))[..., :3]
                    else:
                        try:
                            prev_original_image = cv2.cvtColor(cv2.imread(frame_prev_path), cv2.COLOR_BGR2RGB)
                        except:
                            frame_prev_path = frame_prev_path.replace('gdrive', 'drive') # original files save with path 'content/gdrive/ ...'
                            # print(f'DEBUG: {image_path}')
                            prev_original_image = cv2.cvtColor(cv2.imread(frame_prev_path), cv2.COLOR_BGR2RGB)
                    inputs = self.transform(prev_original_image)  # [:3]
                    prev_frames.append(inputs)
                
        data = {
            'path': image_path,
            'original_image': original_image,
            'inputs': inputs,
            'point2d': point2d,
            'point3d': point3d,
            'mesh2d': mesh2d,
            'bb': bb,
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'keypoints3d': keypoints3d,
            'mesh3d': mesh3d,
            'palm': torch.Tensor(palm[np.newaxis, ...]).float()
        }
        
        if self.IS_MULTIFRAME:
            data['prev_frames'] = prev_frames

        return data

    def __len__(self):
        return len(self.images)
