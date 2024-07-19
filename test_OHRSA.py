from posixpath import split
import os
import torch 
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import argparse

# matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

from utils.dataset import Dataset
from utils.vis_utils import *
from tqdm import tqdm
from models.thor_net import create_thor
from utils.utils import *
# for H2O dataset only
# from utils.h2o_utils.h2o_dataset_utils import load_tar_split
# from utils.h2o_utils.h2o_preprocessing_utils import MyPreprocessor
from utils.options import parse_args_function

import warnings
warnings.filterwarnings('ignore')

### DEBUG time
# from utils.utils_shared import log_time_file_path
# import datetime

# with open(log_time_file_path, 'w') as file:
#     file.write(f'Logging timing for THOR-Net, model-18 checkpoint trained on HO-3D, using 1 GPU ({datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")})\n\n')
#     file.write('-'*50)
#     file.write('\n\n')
### DEBUG time  

'------------------ OTHER INPUT PARAMETERS ------------------'
IS_SAMPLE_DATASET = False # to use a sample of original dataset
TRAINING_SUBSET_SIZE = 100
VALIDATION_SUBSET_SIZE = 10
'------------------------------------------------------------'
'------------------ INPUT PARAMETERS for MULTI-FRAME features ------------------'
N_PREVIOUS_FRAMES = 1
STRIDE_PREVIOUS_FRAMES = 3
'-------------------------------------------------------------------------------'

# Input parameters
args = parse_args_function()

# DEBUG
args.testing = True
args.dataset_name = 'ho3d' # TEST_DATASET, povsurgery, ho3d
args.root = '/content/drive/MyDrive/Thesis/THOR-Net_based_work/ho3d'#'/content/drive/MyDrive/Thesis/THOR-Net_based_work/povsurgery/object_False' 
args.checkpoint_model = '/content/drive/MyDrive/Thesis/THOR-Net_based_work/checkpoints/THOR-Net_trained_on_HO3D/model-18.pkl'
args.mano_root = '/content/drive/MyDrive/Thesis/mano_v1_2/models'
args.obj_root = '/content/THOR-Net/datasets/objects/mesh_1000/book.obj'
args.split = 'test'
args.seq = ''#'d_diskplacer_1/00145'
args.output_results = '/content/drive/MyDrive/Thesis/THOR-Net_based_work/output_results'
args.gpu_number = 0
args.batch_size = 1
args.hid_size = 96
args.photometric = True
args.hands_connectivity_type = 'base'
args.visualize = False
args.object = True

is_evaluate = False
    
other_params = {
    'IS_SAMPLE_DATASET': IS_SAMPLE_DATASET,
    'TRAINING_SUBSET_SIZE': TRAINING_SUBSET_SIZE,
    'VALIDATION_SUBSET_SIZE': VALIDATION_SUBSET_SIZE,
    'IS_MULTIFRAME': False,
    'N_PREVIOUS_FRAMES': N_PREVIOUS_FRAMES,
    'STRIDE_PREVIOUS_FRAMES': STRIDE_PREVIOUS_FRAMES
}    

print(f'args:')
for arg, value in vars(args).items():
    print(f"{arg}: {value}", end=' | ')
print('\n')

left_hand_faces, right_hand_faces, obj_faces = load_faces(mano_root=args.mano_root, obj_root=args.obj_root)

def visualize2d(img, predictions, labels=None, filename=None, palm=None, evaluate=False):
    
    fig = plt.figure(figsize=(20, 10))
    
    H = 1
    if evaluate:
        H = 2
    W = 3

    plot_id = 1
    fig_config = (fig, H, W)
    idx = list(predictions['labels']).index(1) #[0]
    # Plot GT bounding boxes
    if evaluate:
        plot_bb_ax(img, labels, fig_config, plot_id, 'GT BB')
        plot_id += 1
        
        # Plot GT 2D keypoints
        plot_pose2d(img, labels, 0, palm, fig_config, plot_id, 'GT 2D pose')
        plot_id += 1
        
        # Plot GT 3D Keypoints
        plot_pose3d(labels, fig_config, plot_id, 'GT 3D pose', center=palm)
        plot_id += 1

        # Plot GT 3D mesh
        plot_mesh3d(labels, right_hand_faces, obj_faces, fig_config, plot_id, 'GT 3D mesh', center=palm, left_hand_faces=left_hand_faces)
        plot_id += 1

        # Save textured mesh
        texture = generate_gt_texture(img, labels['mesh3d'][0][:, :3])
        save_mesh(labels, filename, right_hand_faces, obj_faces, texture=texture, shape_dir='mesh_gt', left_hand_faces=left_hand_faces)

    # Plot predicted bounding boxes
    plot_bb_ax(img, predictions, fig_config, plot_id, 'RGB frame and Bounding box')
    plot_id += 1

    # Plot predicted 2D keypoints
    plot_pose2d(img, predictions, idx, palm, fig_config, plot_id, 'Predicted 2D pose')
    plot_id += 1

    # plot_pose_heatmap(img, predictions, idx, palm, fig_config, plot_id)
    # plot_id += 1

    # Plot predicted 3D keypoints
    plot_pose3d(predictions, fig_config, plot_id, '3D pose', center=palm)
    plot_id += 1

    # Plot predicted 3D Mesh
    plot_mesh3d(predictions, right_hand_faces, obj_faces, fig_config, plot_id, '3D mesh', center=palm, left_hand_faces=left_hand_faces)
    plot_id += 1

    # Save textured mesh
    predicted_texture = predictions['mesh3d'][0][:, 3:]
    save_mesh(predictions, filename, right_hand_faces, obj_faces, texture=predicted_texture, left_hand_faces=left_hand_faces)
    
    fig.tight_layout()
    plt.show()
    plt.savefig(filename)
    # plt.clf()
    plt.close(fig)

# Transformer function
transform_function = transforms.Compose([transforms.ToTensor()])

num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object)

# Create Output directory

# Dataloader

if args.dataset_name == 'h2o':
    
    h2o_data_dir = os.path.join(args.root, 'shards')
    annotation_components = ['cam_pose', 'hand_pose', 'hand_pose_mano', 'obj_pose', 'obj_pose_rt', 'action_label', 'verb_label']
    # Initialize preprocessor
    my_preprocessor = MyPreprocessor('../mano_v1_2/models/', '../datasets/objects/mesh_1000/', args.root)
    # Dataloader
    input_tar_lists, annotation_tar_files = load_tar_split(h2o_data_dir, args.split)   
    datapipe = create_datapipe(input_tar_lists, annotation_tar_files, annotation_components, args.buffer_size)
    datapipe = datapipe.map(fn=my_preprocessor)
    testloader = torch.utils.data.DataLoader(datapipe, batch_size=args.batch_size, num_workers=2, shuffle=True)
    num_classes = 4
    graph_input='coords'
elif args.dataset_name == 'TEST_DATASET': # DEBUG
    print(f'🟠 Using custom test dataset,', end=' ')
    seq = 'd_diskplacer_1/00145'
    split = 'train'
    print(f'Loading TEST_DATASET data ({seq}) ...')
    testset = Dataset(root=args.root, load_set=args.split, transform=transform_function, num_kps3d=num_kps3d, num_verts=num_verts, other_params=other_params)
    pbar = tqdm(total=len(testset))
    for i, x in enumerate(testset):
        if seq in x['path']:
            print(f'seq {seq} found')
            indices = [i]
            pbar.close()
            break
        else:
            pbar.update(1)
    print(f'Index for "{seq}" found ...', end=' ')
    testset = Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=ho3d_collate_fn)
    num_classes = 2
    graph_input='heatmaps'
    print(f'✅ TEST_DATASET data loaded.')
else:
    print(f'Loading evaluation data ...', end=' ')
    testset = Dataset(root=args.root, load_set=args.split, transform=transform_function, num_kps3d=num_kps3d, num_verts=num_verts, other_params=other_params)
    if IS_SAMPLE_DATASET:
        print('Sub-dataset creation ...', end=' ')
        subset_size = 10
        indices = list(range(subset_size))
        testset = Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=ho3d_collate_fn)
    num_classes = 2
    graph_input='heatmaps'
    print(f'✅ Evaluation data loaded.')
    
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

# Define device
try:
    device = torch.device(f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu')
except:
    args.gpu_number = 0
    device = torch.device(f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu')

# Define model
torch.cuda.empty_cache()
model = create_thor(pretrained=False, num_classes=num_classes, device=device,
                                num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_verts=num_verts,
                                rpn_post_nms_top_n_test=num_classes-1, 
                                box_score_thresh=0.0,
                                photometric=args.photometric, graph_input=graph_input, dataset_name=args.dataset_name,
                                num_features=args.num_features, hid_size=args.hid_size, testing=args.testing,
                                hands_connectivity_type=args.hands_connectivity_type)

if torch.cuda.is_available():
    model = model.cuda(device=args.gpu_number)
    model = nn.DataParallel(model, device_ids=[args.gpu_number])  
        

### Load model
pretrained_model = args.checkpoint_model

# adjust key names, they are in wrong format
state_dict = torch.load(pretrained_model, map_location=device)
try:
    model.load_state_dict(state_dict)
except:
    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)

model = model.eval()
# print(model)
print(f'🟢 Model "{pretrained_model.split(os.sep)[-2]}{os.sep}{pretrained_model.split(os.sep)[-1]}" loaded')

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d']
if args.dataset_name == 'ho3d':
    keys.append('palm')
c = 0
# supporting_dicts = (pickle.load(open('./rcnn_outputs/rcnn_outputs_778_test_3d.pkl', 'rb')),
#                     pickle.load(open('./rcnn_outputs_mesh/rcnn_outputs_778_test_3d.pkl', 'rb')))
supporting_dicts = None
output_dicts = ({}, {})

evaluate = False
errors = [[], [], [], [], [], []]
# if args.split == 'test' or (args.dataset_name == 'h2o' and args.split == 'test'):  
if is_evaluate:
    evaluate = True

# rgb_errors = []

for i, ts_data in tqdm(enumerate(testloader), total=len(testloader), desc='Evaluation: '):
    
    data_dict = ts_data
    path = data_dict[0]['path'].split(os.sep)[-1]
    if args.dataset_name=='ho3d' or args.dataset_name=='TEST_DATASET': # choose specific sequence to evaluate
        if args.seq not in data_dict[0]['path']:
            continue
        if '_' in path:
            path = path.split('_')[-1]
        # frame_num = int(path.split('.')[0])
    elif args.dataset_name=='povsurgery':
        seq_name = data_dict[0]['path'].split(os.sep)[-2]
    else:
        pass
        
    ### Run inference
    inputs = [t['inputs'].to(device) for t in data_dict]
    
    # # DEBUG time
    # with open(log_time_file_path, 'a') as file:
    #     file.write(f'{datetime.datetime.now()} | START Inputs {i+1}\n')
    _, result = model(inputs)
    # with open(log_time_file_path, 'a') as file:
    #     file.write(f'{datetime.datetime.now()} | END Inputs {i+1}\n')
    outputs = (result, _)
    img = inputs[0].cpu().detach().numpy()
    
    # predictions, img, palm, labels = prepare_data_for_evaluation(data_dict, outputs, img, keys, device, args.split) DEBUG

    ### Visualization
    if args.visualize: 

        name = path.split('/')[-1]
        output_dir = os.path.join(args.output_results, seq)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if (num_classes == 2 and 1 in predictions['labels']) or (num_classes == 4 and set([1, 2, 3]).issubset(predictions['labels'])):
            visualize2d(img, predictions, labels, filename=f'{os.path.join(output_dir, name)}', palm=palm, evaluate=evaluate)
        else:
            cv2.imwrite(f'{os.path.join(output_dir, name)}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ### Evaluation
    if evaluate:
        c = save_calculate_error(predictions, labels, path, errors, output_dicts, c, num_classes, args.dataset_name, obj=args.object, generate_mesh=True)
    # if i == 10:
    #     break


if evaluate:
    names = ['lh pose', 'lh mesh', 'rh pose', 'rh mesh', 'obj pose', 'obj mesh']

    for i in range(len(errors)):
        avg_error = np.average(np.array(errors[i]))
        print(f'{names[i]} average error on test set:', avg_error)

    # avg_error = np.average(np.array(errors))
    # print('Hand shape average error on validation set:', avg_error)

    # avg_rgb_error = np.average(np.array(rgb_errors))
    # print('Texture average error on validation set:', avg_rgb_error)

# save_dicts(output_dicts, args.split)