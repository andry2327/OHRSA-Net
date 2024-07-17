import os
import numpy as np
import trimesh
import pickle
import torch
import torchvision.transforms as transforms
import cv2
import os
import argparse
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append('THOR-Net/datasets/pov_surgery_utils')
from pov_surgery_utils.utils.manopth.manopth.manolayer import ManoLayer
from sklearn.preprocessing import MinMaxScaler
import joblib
from pov_surgery_utils.pov_surgery_processing import POVSURGERY
from pov_surgery_utils.pov_surgery_dataset_split import PovSurgerySplits
from collections import defaultdict

# Input parameters
parser = argparse.ArgumentParser()

# Loading dataset    
parser.add_argument("--root", required=True, help="HO3D dataset folder")
parser.add_argument("--mano_root", required=True, help="Path to MANO models")
parser.add_argument("--YCBModelsDir", default='./datasets/ycb_models', help="Path to YCB object meshes folder")
parser.add_argument("--dataset_path", default='./datasets/ho3d', help="Where to store dataset files")
parser.add_argument("--object", action='store_true', help="Generate 3D pose or mesh for the object")

args = parser.parse_args()

root = args.root
YCBModelsDir = args.YCBModelsDir
dataset_path = args.dataset_path
mano_root = args.mano_root
is_object = args.object

print(f'is_object = {is_object}')

BASE_DATA_FILES_PATH = '/content/drive/MyDrive/Thesis/POV_Surgery/data'

# DEBUG
# root = '/content/drive/MyDrive/Thesis/POV_Surgery_data'
# # YCBModelsDir = args.YCBModelsDir
# dataset_path = '/content/drive/MyDrive/Thesis/THOR-Net_based_work/povsurgery'
# mano_root = '/content/drive/MyDrive/Thesis/mano_v1_2/models'
# is_object = True

# Get original POV-Surgery splits 
train_list, test_list = PovSurgerySplits().get_splits()
val_list = ["d_scalpel_1", "r_scalpel_3", "r_diskplacer_5", "s_friem_2", "s_scalpel_3"]

# Load object mesh
reorder_idx = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

# Train
base_info_train = pickle.load(open(os.path.join(root, 'handoccnet_train/2d_repro_ho3d_style_hocc_cleaned.pkl'), 'rb'))
set_list_train = list(base_info_train.keys())
# Validation:
base_info_validation = pickle.load(open(os.path.join(root, 'handoccnet_train/2d_repro_ho3d_style_hocc_cleaned.pkl'), 'rb'))
set_list_validation = list(base_info_validation.keys())
# Evaluation
base_info_evaluation = pickle.load(open(os.path.join(root, 'handoccnet_train/2d_repro_ho3d_style_test_cleaned.pkl'), 'rb'))
set_list_evaluation = list(base_info_evaluation.keys())
evaluation_list = list(set([x.split('/')[0] for x in set_list_evaluation]))

# Get info about start and end frame for each dataset entry
frames_info_dict = defaultdict(list)
for key in {**base_info_train, **base_info_evaluation}.keys(): 
    file_name, frame = key.split('/')
    frames_info_dict[file_name].append(frame)
for k, v in frames_info_dict.items():
    frames_info_dict[k] = (min(v), max(v))

# for k, v in frames_info_dict.items(): # DEBUG
#     print(f'{k}: {v}') # DEBUG

# # DEBUG
# val_list = ["d_scalpel_1", 'r_diskplacer_5', 'i_friem_2'] # DEBUG
# train_list = ['s_diskplacer_2', 'm_scalpel_1']
# train_list.extend(val_list) # DEBUG
# evaluation_list = ['i_scalpel_1', 'r_friem_3', 'R2_s_diskplacer_1'] # DEBUG

print()
print('-'*30)
print(f'Dataset length: {len(PovSurgerySplits().DATASET_ENTRIES_NAMES)} items')
print()
print(f'train_list (with val_list) - {len(train_list)} items:')
print(train_list)
print(f'    val_list - {len(val_list)} items:')
print(f'    {val_list}')
print(f'evaluation_list - {len(evaluation_list)} items:')
print(evaluation_list)
print('-'*30)
print()

def save_as_npy(file_path, data):
    np.save(file_path, np.array(data, dtype=object))

def save_as_pkl(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def fit_scaler(arr, k):

    scaler = MinMaxScaler()
    scaled = scaler.fit(arr)
    print(f'{k} scaler min:', scaler.data_min_, ', scaler max:', scaler.data_max_)
    joblib.dump(scaler, f'{k}_scaler.save') 

    return scaler

def normalize(arr, normalizer):

    all_points = arr.reshape((-1, arr.shape[-1]))
    normalized_points = normalizer.transform(all_points)

    return normalized_points.reshape(arr.shape)

class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'f': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])

    for k, v in d.items():
        if k in ['v','f']:
            if v:
                d[k] = np.vstack(v)
            else:
                print(k)
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result

obj_dict = {}
def load_ycb_obj(ycb_dataset_path, obj_name, rot=None, trans=None, decimationValue=1000):
    ''' Load a YCB mesh based on the name '''
    if obj_name not in obj_dict.keys():
        path = os.path.join(ycb_dataset_path, obj_name, f'morphed_sphere_{decimationValue}.obj')
        obj_mesh = read_obj(path)
        obj_dict[obj_name] = obj_mesh        
    else:
        obj_mesh = obj_dict[obj_name]
    # apply current pose to the object model
    if rot is not None:
        obj_mesh_verts = np.matmul(obj_mesh.v, cv2.Rodrigues(rot)[0].T) + trans

    # Change to non openGL coords and convert from m to mm
    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    obj_mesh_verts = obj_mesh_verts.dot(coordChangeMat.T) * 1000
        
    return obj_mesh_verts, obj_mesh.f

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts

def load_mesh_from_manolayer(fullpose, beta, trans, mano_layer):
    
    # Convert inputs to tensors and reshape them to be compatible with Mano Layer
    fullpose_tensor = torch.as_tensor(fullpose, dtype=torch.float32).reshape(1, -1)
    shape_tensor = torch.as_tensor(beta, dtype=torch.float32).reshape(1, -1)
    trans_tensor = torch.as_tensor(trans, dtype=torch.float32).reshape(1, -1)

    # Pass to Mano layer
    hand_verts, hand_joints = mano_layer(fullpose_tensor, shape_tensor, trans_tensor)
    
    # return outputs as numpy arrays and scale them back from mm to m 
    hand_verts = hand_verts.cpu().detach().numpy()[0] 
    hand_joints = hand_joints.cpu().detach().numpy()[0]

    return hand_joints, hand_verts, mano_layer.th_faces

def transform_annotations(data, mano_layer, subset='train'):
    return data
    # return hand_object3d, hand_object2d, mesh3d, mesh2d
    
import numpy as np
import trimesh
import scipy.io as sio

def compute_3d_object_corners(annotations, object_type):
    
    SCALPEL_OFFSET = [0.04805371, 0, 0]
    DISKPLACER_OFFSET = [0, 0.34612157, 0]
    FRIEM_OFFSET = [0, 0.1145, 0]

    object_file = os.path.join(BASE_DATA_FILES_PATH, 'tool_mesh', object_type + '.stl')
    control_point_file = os.path.join(BASE_DATA_FILES_PATH, 'tool_mesh', 'tool_control_points.mat')
    
    mesh_object = trimesh.load(object_file)
    control_point_mat = sio.loadmat(control_point_file)
    
    if 'diskplacer' in object_type:
        mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(DISKPLACER_OFFSET)
        tool_control_point = control_point_mat['diskplacer_kp'] * 0.001 - np.array(DISKPLACER_OFFSET)
    elif 'friem' in object_type:
        mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(FRIEM_OFFSET)
        tool_control_point = control_point_mat['friem_kp'] * 0.001 - np.array(FRIEM_OFFSET)
    elif 'scalpel' in object_type:
        mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(SCALPEL_OFFSET)
        tool_control_point = control_point_mat['scalpel_kp'] * 0.001 - np.array(SCALPEL_OFFSET)

    base_object_rot = annotations['base_object_rot']
    grab2world_R = annotations['grab2world_R']
    grab2world_T = annotations['grab2world_T']
    
    # Apply the base object rotation
    mesh_object.vertices = mesh_object.vertices @ base_object_rot.T
    
    # Apply the grab to world rotation and translation
    mesh_object.vertices = mesh_object.vertices @ grab2world_R + grab2world_T
    
    # Extract the corners of the bounding box
    bbox = mesh_object.bounding_box.bounds
    corners = np.array([
        [bbox[0, 0], bbox[0, 1], bbox[0, 2]],
        [bbox[0, 0], bbox[0, 1], bbox[1, 2]],
        [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
        [bbox[0, 0], bbox[1, 1], bbox[1, 2]],
        [bbox[1, 0], bbox[0, 1], bbox[0, 2]],
        [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
        [bbox[1, 0], bbox[1, 1], bbox[0, 2]],
        [bbox[1, 0], bbox[1, 1], bbox[1, 2]]
    ])
    
    return corners

def load_annotations(data, mano_layer, subset='train'):
    
    hand_joints, hand_mesh3d, _ = load_mesh_from_manolayer(np.concatenate((data['mano']['global_orient'], data['mano']['hand_pose']), axis=1), data['mano']['betas'], data['mano']['transl'], mano_layer)
    
    cam_intr = data['cam_intr']

    data['handJoints3D'] = hand_joints
    if subset == 'train':
        hand3d = data['handJoints3D'][reorder_idx] 
    else:
        # hand3d = data['handJoints3D'].reshape((1, -1)) # TODO: to check
        hand3d = data['handJoints3D']

    if 'diskplacer' in data['seqName']: 
        object_type = 'diskplacer'
    elif 'friem' in data['seqName']: 
        object_type = 'friem'
    else: 
        object_type = 'scalpel'
            
    obj_corners = compute_3d_object_corners(data, object_type) # shape=(8, 3)
    # print(data)
    # print(len(data['handBoundingBox']))
    # Convert to non-OpenGL coordinates and multiply by thousand to convert from m to mm
    if is_object:
        hand_object3d = np.concatenate([hand3d, obj_corners]) * 1000
    else:
        hand_object3d = hand3d * 1000
    # hand_object3d = hand_object3d.dot(coordChangeMat.T)

    # Project from 3D world to Camera coordinates using the camera matrix  
    hand_object3d = hand_object3d.dot(coordChangeMat.T) 
    hand_object2d = np.array([])
    hand_object_proj = cam_intr.dot(hand_object3d.transpose()).transpose()
    hand_object2d = (hand_object_proj / hand_object_proj[:, 2:])[:, :2]

    mesh3d = np.array([])
    mesh2d = np.array([])
    
    if subset == 'train':
        
        # Project from 3D world to Camera coordinates using the camera matrix    
        hand_mesh3d = hand_mesh3d.dot(coordChangeMat.T)
        hand_mesh2d = np.array([])
        hand_mesh_proj = cam_intr.dot(hand_mesh3d.transpose()).transpose()
        hand_mesh2d = (hand_mesh_proj / hand_mesh_proj[:, 2:])[:, :2]

        # Do the same for the object
        '''obj_mesh3d, _ = load_ycb_obj(YCBModelsDir, data['objName'], data['objRot'], data['objTrans'])
        obj_mesh_proj = cam_intr.dot(obj_mesh3d.transpose()).transpose()
        obj_mesh2d = (obj_mesh_proj / obj_mesh_proj[:, 2:])[:, :2]        
        
        mesh3d = np.concatenate((hand_mesh3d, obj_mesh3d), axis=0)
        mesh2d = np.concatenate((hand_mesh2d, obj_mesh2d), axis=0) '''
        
        mesh3d = hand_mesh3d
        mesh2d = hand_mesh2d
    
    return hand_object3d, hand_object2d, mesh3d, mesh2d

if __name__ == '__main__':

    mano_layer = ManoLayer(mano_root=mano_root, use_pca=False, ncomps=45, flat_hand_mean=True)
    names = ['images', 'depths', 'points2d', 'points3d', 'mesh3d', 'mesh2d']
    file_dict_train = defaultdict(list)
    file_dict_val = defaultdict(list)
    name_object_dict = {}

    directory = f'val_size_{len(val_list)}'
    if not os.path.exists(directory):
        os.makedirs(directory)


    # # Training
    print('Processing train-validation splits:')
    dataset = POVSURGERY(transforms.ToTensor(), "train")
    
    # Progress bar
    total = 0
    slice = []
    for subject in train_list:
        rgb = os.path.join(root, 'color', subject)
        rgb_listdir_filtered = [x for x in sorted(os.listdir(rgb)) if frames_info_dict[subject][0] <= x <= frames_info_dict[subject][1]]
        # slice = [int(x) for x in np.linspace(0, len(rgb_listdir_filtered)-1, 10)] # DEBUG
        # total += len(np.array(rgb_listdir_filtered)[slice]) # DEBUG
        total += len(rgb_listdir_filtered)
        
    pbar = tqdm(total=total)
    error_count = 0
    error_data_extended_count = 0
    for subject in sorted(train_list):
        rgb = os.path.join(root, 'color', subject)
        depth = os.path.join(root, 'depth', subject)
        meta = os.path.join(root, 'annotation', subject)    
        
        # for each dataset entry, only use frames with annotations
        rgb_listdir_filtered = [x for x in sorted(os.listdir(rgb)) if frames_info_dict[subject][0] <= x <= frames_info_dict[subject][1]]
        # slice = [int(x) for x in np.linspace(0, len(rgb_listdir_filtered)-1, 10)] # DEBUG
        for rgb_file in sorted(rgb_listdir_filtered):
            file_number = rgb_file.split('.')[0]
            # Error in POV_SURGERY: some entries misses initial frame 00000
            # -> copied from 00001 entries
            file_number_meta_fixed = file_number if file_number!='00000' else '00001'
            seqName_id = f'{subject}/{file_number_meta_fixed}'
            try:
                data_extended = dataset.get_item(subject, file_number_meta_fixed) # Load additional data from POV-Surgery annotations
            except:
                # print(f'🔴 Error data_extended: {subject, file_number_meta_fixed}')
                error_data_extended_count += 1
            meta_file = os.path.join(meta, file_number_meta_fixed+'.pkl')
            img_path = os.path.join(rgb, rgb_file)        
            depth_path = os.path.join(depth, file_number+'.png')        
            
            try:
                data = np.load(meta_file, allow_pickle=True)
            except:
                #file error, copied from previous frame
                error_count += 1
            # except:
            #     print(f'🟠 Problem with file {meta_file}, file skipped')
            #     count += 1

            if 'handJoints3D' in data and data['handJoints3D'] is None:
                # Load previous frame's data if data is missing
                # count += 1
                continue
            else:
                data = {**data, **data_extended[0], **data_extended[1], **data_extended[2]} # extend data with additional annotations
                # data = transform_annotations(data, mano_layer) # make them compatible with HO-3D style and fields needed
                hand_object3d, hand_object2d, mesh3d, mesh2d = load_annotations(data, mano_layer)

            values = [img_path, depth_path, hand_object2d, hand_object3d, mesh3d, mesh2d]
            
            if subject in val_list:
                for i, name in enumerate(names):
                    file_dict_val[name].append(values[i])
            else:
                for i, name in enumerate(names):
                    file_dict_train[name].append(values[i])
            
            pbar.update(1)
    pbar.close()

    # print('Total number of failures:', count)
    print("`Size of training dataset", len(file_dict_train['points2d']))
    print("Size of validation dataset", len(file_dict_val['points2d']))
    print(f"# errors: {error_count}/{len(file_dict_train['points2d'])} ({error_count/len(file_dict_train['points2d']):.2%})")
    print(f"# errors frame annotations (data_extended): {error_data_extended_count}/{total} ({error_data_extended_count/total:.2%})")

    # Appending all possible 2D points to normalize
    # points_2d_lists = [file_dict_train['hand_mesh2d'], file_dict_train['points2d'], file_dict_val['hand_mesh2d'], file_dict_val['points2d']]
    # points_2d_reshaped = []
    # for l in points_2d_lists:
    #     reshaped_arr = np.array(l).reshape((-1, 2))
    #     points_2d_reshaped.append(reshaped_arr)
    # all_points2d = np.concatenate(points_2d_reshaped)

    # # Create a scaler object to normalize 2D points
    # scaler2d = fit_scaler(all_points2d, '2d')    

    # # Appending all possible 3D points to normalize
    # points_3d_lists = [file_dict_train['hand_mesh'], file_dict_train['points3d'], file_dict_val['hand_mesh'], file_dict_val['points3d']]
    # points_3d_reshaped = []
    # for l in points_3d_lists:
    #     reshaped_arr = np.array(l).reshape((-1, 3))
    #     points_3d_reshaped.append(reshaped_arr)
    # all_points3d = np.concatenate(points_3d_reshaped)
    
    # # Create a scaler object to normalize 3D points
    # scaler3d = fit_scaler(all_points3d, '3d')


    for k, v in file_dict_train.items():
        npy_path = f'{dataset_path}/{k}-train.npy'
        pkl_path = f'{dataset_path}/{k}-train.pkl'
        try:
            save_as_npy(npy_path, v)
            print(f'🟢 SAVED {npy_path}: shape={np.array(v).shape}')
        except Exception as e:
            print(f'🔴 ERROR saving {npy_path} as .npy: {e}')
            try:
                save_as_pkl(pkl_path, v)
                print(f'🟢 SAVED {pkl_path}')
            except Exception as e:
                print(f'🔴 ERROR saving {pkl_path} as .pkl: {e}')

    for k, v in file_dict_val.items():
        npy_path = f'{dataset_path}/{k}-val.npy'
        pkl_path = f'{dataset_path}/{k}-val.pkl'
        try:
            save_as_npy(npy_path, v)
            print(f'🟢 SAVED {npy_path}: shape={np.array(v).shape}')
        except Exception as e:
            print(f'🔴 ERROR saving {npy_path} as .npy: {e}')
            try:
                save_as_pkl(pkl_path, v)
                print(f'🟢 SAVED {pkl_path}')
            except Exception as e:
                print(f'🔴 ERROR saving {pkl_path} as .pkl: {e}')

    print()
    file_dict_test = defaultdict(list)
    name_object_dict = {}

    # # Evaluation
    # count = 0
    print('Processing evaluation split:')
    dataset = POVSURGERY(transforms.ToTensor(), "evaluation")
    
    # Progress bar
    total = 0
    slice = []
    for subject in evaluation_list:
        rgb = os.path.join(root, 'color', subject)
        rgb_listdir_filtered = [x for x in sorted(os.listdir(rgb)) if frames_info_dict[subject][0] <= x <= frames_info_dict[subject][1]]
        # slice = [int(x) for x in np.linspace(0, len(rgb_listdir_filtered)-1, 10)] 
        # total += len(np.array(rgb_listdir_filtered)[slice]) 
        total += len(rgb_listdir_filtered)
        
    pbar = tqdm(total=total)
    error_count = 0
    error_data_extended_count = 0
    for subject in sorted(evaluation_list):
        rgb = os.path.join(root, 'color', subject)
        depth = os.path.join(root, 'depth', subject)
        meta = os.path.join(root, 'annotation', subject)
        
        # for each dataset entry, only use frames with annotations
        rgb_listdir_filtered = [x for x in sorted(os.listdir(rgb)) if frames_info_dict[subject][0] <= x <= frames_info_dict[subject][1]]
        # slice = [int(x) for x in np.linspace(0, len(rgb_listdir_filtered)-1, 10)] # DEBUG
        for rgb_file in sorted(rgb_listdir_filtered): 
            file_number = rgb_file.split('.')[0]
            # Error in POV_SURGERY: some entries misses initial frame 00000
            # -> copied from 00001 entries
            file_number_meta_fixed = file_number if file_number!='00000' else '00001'
            seqName_id = f'{subject}/{file_number_meta_fixed}'
            try:
                data_extended = dataset.get_item(subject, file_number_meta_fixed) # Load additional data from POV-Surgery annotations
            except:
                # print(f'🔴 Error data_extended: {subject, file_number_meta_fixed}')
                error_data_extended_count += 1 
            meta_file = os.path.join(meta, file_number_meta_fixed+'.pkl')
            img_path = os.path.join(rgb, rgb_file)
            depth_path = os.path.join(depth, file_number+'.png')  
            
            try:
                data = np.load(meta_file, allow_pickle=True)
            except:
                error_count += 1
                # print(f'🟠 Problem with file {meta_file}, file skipped')
                
            if 'handJoints3D' in data and data['handJoints3D'] is None:
                continue
                # hand_object3d, hand_object2d, mesh3d, mesh2d = last_hand_object3d, last_hand_object2d, last_mesh3d, last_mesh2d
            else:
                data = {**data, **data_extended[0], **data_extended[1], **data_extended[2]} # extend data with additional annotations
                # data = transform_annotations(data, mano_layer) # make them compatible with HO-3D style and fields needed
                hand_object3d, hand_object2d, mesh3d, mesh2d = load_annotations(data, mano_layer, subset='test')
                # last_hand_object3d, last_hand_object2d, last_mesh3d, last_mesh2d = hand_object3d, hand_object2d, mesh3d, mesh2d
                # print(hand_object3d.shape, hand_object2d.shape, mesh3d.shape, mesh2d.shape)
      
            values = [img_path, depth_path, hand_object2d, hand_object3d, mesh3d, mesh2d]
            
            for i, name in enumerate(names):
                file_dict_test[name].append(values[i])
                
            pbar.update(1)
    pbar.close()

    print("Size of testing dataset", len(file_dict_test['points2d']))
    print(f"# errors: {error_count} ({error_count/len(file_dict_test['points2d']):.2%})")
    print(f"# errors frame annotations (data_extended): {error_data_extended_count}/{total} ({error_data_extended_count/total:.2%})")
    # print("total testing samples:", count, "percentage:", len(file_dict_test['points2d'])/count)
    
    for k, v in file_dict_test.items():
        npy_path = f'{dataset_path}/{k}-test.npy'
        pkl_path = f'{dataset_path}/{k}-test.pkl'
        try:
            save_as_npy(npy_path, v)    
            print(f'🟢 SAVED {npy_path}: shape={np.array(v).shape}')
        except Exception as e:
            print(f'🔴 ERROR saving {npy_path} as .npy: {e}')
            try:
                save_as_pkl(pkl_path, v)
                print(f'🟢 SAVED {pkl_path}')
            except Exception as e:
                print(f'🔴 ERROR saving {pkl_path} as .pkl: {e}')