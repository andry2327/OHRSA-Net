import torch
import numpy as np
import pickle
import torchvision.transforms as transforms
# for H2O dataset only
# from .h2o_utils.h2o_datapipe_pt_1_12 import create_datapipe
from .dataset import Dataset
from torch.utils.data import Subset
from tqdm import tqdm
import os

def ho3d_collate_fn(batch):
    # print(batch, '\n--------------------\n')
    # print(len(batch))
    return batch

def h2o_collate_fn(samples):
    output_list = []
    for sample in samples:
        sample_dict = {
            'path': sample[0],
            'inputs': sample[1],
            'keypoints2d': sample[2],
            'keypoints3d': sample[3].unsqueeze(0),
            'mesh2d': sample[4],
            'mesh3d': sample[5].unsqueeze(0),
            'boxes': sample[6],
            'labels': sample[7],
            'keypoints': sample[8]
        }
        output_list.append(sample_dict)
    return output_list

def create_loader(dataset_name, root, split, batch_size, num_kps3d=21, num_verts=778, h2o_info=None, other_params={}):

    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name.lower() == 'h2o':
        input_tar_lists, annotation_tar_files, annotation_components, shuffle_buffer_size, my_preprocessor = h2o_info
        datapipe = create_datapipe(input_tar_lists, annotation_tar_files, annotation_components, shuffle_buffer_size)
        datapipe = datapipe.map(fn=my_preprocessor)
        loader = torch.utils.data.DataLoader(datapipe, batch_size=batch_size, num_workers=2, shuffle=True)
    elif dataset_name == 'TEST_DATASET':
        seq = 'd_diskplacer_1/00145'
        print(f'Using custom train-val dataset ({seq}) ...', end=' ')
        split = 'train'
        dataset = Dataset(root=root, load_set=split, transform=transform, num_kps3d=num_kps3d, num_verts=num_verts, other_params=other_params)
        print(f'dataset loaded ({seq}) ...', end=' ')
        pbar = tqdm(total=len(dataset))
        for i, x in enumerate(dataset):
            if seq in x['path']:
                print(f'seq {seq} found')
                indices = [i]
                pbar.close()
                break
            else:
                pbar.update(1)
        print(f'ind for "{seq}" found ...', end=' ')
        dataset = Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=ho3d_collate_fn)
    else:
        dataset = Dataset(root=root, load_set=split, transform=transform, num_kps3d=num_kps3d, num_verts=num_verts, other_params=other_params)    
        if other_params['IS_SAMPLE_DATASET']:
            print('Sub-dataset creation ...', end=' ')
            subset_size = other_params['TRAINING_SUBSET_SIZE'] if split=='train' else other_params['VALIDATION_SUBSET_SIZE']
            indices = list(range(subset_size))
            dataset = Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=ho3d_collate_fn)
    return loader

def freeze_component(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
def calculate_keypoints(dataset_name, obj):

    if dataset_name == 'ho3d':
        num_verts = 1778 if obj else 778
        num_kps3d = 29 if obj else 21
        num_kps2d = 29 if obj else 21
    elif dataset_name == 'povsurgery' or 'TEST_DATASET':
        num_verts = 778
        num_kps3d = 29 if obj else 21
        num_kps2d = 29 if obj else 21
    else: # h20
        num_verts = 2556 if obj else 1556
        num_kps3d = 50 if obj else 42
        num_kps2d = 21

    return num_kps2d, num_kps3d, num_verts

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def save_calculate_error(predictions, labels, path, errors, output_dicts, c, num_classes=2, dataset_name='h2o', obj=True, generate_mesh=False):
    """Stores the results of the model in a dict and calculates error in case of available gt"""

    predicted_labels = list(predictions['labels'])

    rhi, obji = 0, 21
    rhvi, objvi = 0, 778

    if dataset_name == 'h2o':
        rhi, obji = 21, 42
        rhvi, objvi = 778, 778*2

    if (num_classes > 2 and set([1, 2, 3]).issubset(predicted_labels)) or (num_classes == 2 and 1 in predicted_labels):
        
        keypoints = predictions['keypoints3d'][0]
        keypoints_gt = labels['keypoints3d'][0]
        
        if generate_mesh:
            mesh = predictions['mesh3d'][0][:, :3]
            mesh_gt = labels['mesh3d'][0]
        else:
            mesh = np.zeros((2556, 3))
            mesh_gt = np.zeros((2556, 3))

        rh_pose, rh_pose_gt = keypoints[rhi:rhi+21], keypoints_gt[rhi:rhi+21]
        rh_mesh, rh_mesh_gt = mesh[rhvi:rhvi+778], mesh_gt[rhvi:rhvi+778]

        if obj:
            obj_pose, obj_pose_gt = keypoints[obji:], keypoints_gt[obji:]
            obj_mesh, obj_mesh_gt = mesh[objvi:], mesh_gt[objvi:]
        else:
            obj_pose, obj_pose_gt = np.zeros((8, 3)), np.zeros((8, 3))
            obj_mesh, obj_mesh_gt = np.zeros((1000, 3)), np.zeros((1000, 3))
            

        if dataset_name == 'h2o':
            lh_pose, lh_pose_gt = keypoints[:21], keypoints_gt[:21]
            lh_mesh, lh_mesh_gt = mesh[:778], mesh_gt[:778]
        else:
            lh_pose, lh_pose_gt = np.zeros((21, 3)), np.zeros((21, 3))
            lh_mesh, lh_mesh_gt = np.zeros((778, 3)), np.zeros((778, 3))

        pair_list = [
            (lh_pose, lh_pose_gt),
            (lh_mesh, lh_mesh_gt),
            (rh_pose, rh_pose_gt),
            (rh_mesh, rh_mesh_gt),
            (obj_pose, obj_pose_gt),
            (obj_mesh, obj_mesh_gt)
        ]

        for i in range(len(pair_list)):

            error = mpjpe(torch.Tensor(pair_list[i][0]), torch.Tensor(pair_list[i][1]))
            errors[i].append(error)

        error = mpjpe(torch.Tensor(mesh), torch.Tensor(mesh_gt))
    else:
        c += 1
        error = 1000
        keypoints = np.zeros((50, 3))
        mesh = np.zeros((2556, 3))
        # print(c)
      
    output_dicts[0][path] = keypoints
    output_dicts[1][path] = mesh   

    # Object pose
    # output_dicts[1][path] = keypoints_gt[42:]

    return c

def save_dicts(output_dicts, split):
    
    output_dict = dict(sorted(output_dicts[0].items()))
    output_dict_mesh = dict(sorted(output_dicts[1].items()))
    print('Total number of predictions:', len(output_dict.keys()))

    with open(f'./outputs/rcnn_outputs/rcnn_outputs_21_{split}_3d_v3.pkl', 'wb') as f:
        pickle.dump(output_dict, f)

    with open(f'./outputs/rcnn_outputs/rcnn_outputs_778_{split}_3d_v3.pkl', 'wb') as f:
        pickle.dump(output_dict_mesh, f)

def prepare_data_for_evaluation(data_dict, outputs, img, keys, device, split):
    """Postprocessing function"""

    # print(data_dict[0])
    targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]

    labels = {k: v.cpu().detach().numpy() for k, v in targets[0].items()}
    predictions = {k: v.cpu().detach().numpy() for k, v in outputs[0].items()}

    palm = None
    if 'palm' in labels.keys():
        palm = labels['palm'][0]

    if split == 'test':
        labels = None

    img = img.transpose(1, 2, 0) * 255
    img = np.ascontiguousarray(img, np.uint8) 

    return predictions, img, palm, labels

def project_3D_points(pts3D):

    cam_mat = np.array(
        [[617.343,0,      312.42],
        [0,       617.343,241.42],
        [0,       0,       1]])

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0] / proj_pts[:,2], proj_pts[:,1] / proj_pts[:,2]], axis=1)
    # proj_pts = proj_pts.to(torch.long)
    return proj_pts


def generate_gt_texture(image, mesh3d):
    mesh2d = project_3D_points(mesh3d)

    image = image / 255

    H, W, _ = image.shape

    idx_x = mesh2d[:, 0].clip(min=0, max=W-1).astype(np.int)
    idx_y = mesh2d[:, 1].clip(min=0, max=H-1).astype(np.int)

    texture = image[idx_y, idx_x]
    
    return texture

def calculate_rgb_error(image, mesh3d, p_texture):
    texture = generate_gt_texture(image, mesh3d)
    error = mpjpe(torch.Tensor(texture), torch.Tensor(p_texture))
    return error
