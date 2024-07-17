# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

""" import libraries"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os
import datetime
import pytz
import cv2
from tqdm import tqdm

from utils.options import parse_args_function
from utils.utils import freeze_component, calculate_keypoints, create_loader, prepare_data_for_evaluation

# for H2O dataset only
# from utils.h2o_utils.h2o_dataset_utils import load_tar_split 
# from utils.h2o_utils.h2o_preprocessing_utils import MyPreprocessor

from models.thor_net import create_thor
torch.multiprocessing.set_sharing_strategy('file_system')

'------------------ OTHER INPUT PARAMETERS ------------------'
IS_SAMPLE_DATASET = False # to use a sample of original dataset
TRAINING_SUBSET_SIZE = 100
VALIDATION_SUBSET_SIZE = 10
'------------------------------------------------------------'
'------------------ INPUT PARAMETERS for MULTI-FRAME features ------------------'
N_PREVIOUS_FRAMES = 1
STRIDE_PREVIOUS_FRAMES = 3
'-------------------------------------------------------------------------------'

args = parse_args_function()
output_folder = args.output_file.rpartition(os.sep)[0]
# print(f'args:')
# for arg, value in vars(args).items():
#     print(f"{arg}: {value}", end=' | ')
# print('-'*30)

# DEBUG
args.dataset_name = 'TEST_DATASET' # ho3d, povsurgery, TEST_DATASET
args.root = '/content/drive/MyDrive/Thesis/THOR-Net_based_work/povsurgery/object_False' 
args.output_file = '/content/drive/MyDrive/Thesis/THOR-Net_based_work/checkpoints/THOR-Net_trained_on_POV-Surgery_object_False/Training-KE--01-07-2024_10-27/model-' 
output_folder = args.output_file.rpartition(os.sep)[0]
if not os.path.exists(output_folder):
    os.mkdir(output_folder) 
args.batch_size = 1
args.num_iteration = 20
args.object = False 
args.hid_size = 96
args.photometric = True
args.multiframe = False 
args.log_batch = 1 # frequency to print training losses
args.val_epoch = 1 # frequency to compute validation loss
args.pretrained_model=''#'/content/drive/MyDrive/Thesis/THOR-Net_trained_on_POV-Surgery_object_False/Training-100samples--20-06-2024_17-08/model-22.pkl'
args.hands_connectivity_type = 'base'
# args.visualize = True
# args.output_results = '/content/drive/MyDrive/Thesis/THOR-Net_trained_on_POV-Surgery_object_False/Training-100samples--20-06-2024_17-08/output_results'

other_params = {
    'IS_SAMPLE_DATASET': IS_SAMPLE_DATASET,
    'TRAINING_SUBSET_SIZE': TRAINING_SUBSET_SIZE,
    'VALIDATION_SUBSET_SIZE': VALIDATION_SUBSET_SIZE,
    'IS_MULTIFRAME': args.multiframe,
    'N_PREVIOUS_FRAMES': N_PREVIOUS_FRAMES,
    'STRIDE_PREVIOUS_FRAMES': STRIDE_PREVIOUS_FRAMES
}

# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object)

""" Configure a log """

files_in_dir = os.listdir(output_folder)
log_file = [x for x in files_in_dir if x.endswith('.txt')]

if log_file:
    # If there is an existing log file, use the first one found
    filename_log = os.path.join(output_folder, log_file[0])
    
    # delete all lines referring to higher epochs of trained model
    current_model_epoch = int(args.pretrained_model.split('-')[-1].split('.')[0])
    pattern = f'Epoch {current_model_epoch+1}/'
    with open(filename_log, 'r') as file:
        lines = file.readlines()
    index = None
    for i, line in enumerate(lines):
        if pattern in line:
            index = i
            break
    if index is not None:
        lines = lines[:index]
    with open(filename_log, 'w') as file:
        file.writelines(lines)
else:
    # Create a new log file
    filename_log = os.path.join(
        output_folder,
        f'log_{output_folder.rpartition(os.sep)[-1]}.txt'
    )

# Configure the logging
log_format = '%(message)s'
logging.basicConfig(
    filename=filename_log,
    level=logging.INFO,
    format=log_format,
    filemode='a'  
)
fh = logging.FileHandler(filename_log, mode='a')  # Use 'a' mode to append to the log file
fh.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger(__name__)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

logging.info(f'args:') if not log_file else None
print(f'args:')
for arg, value in vars(args).items():
    logging.info(f"--{arg}: {value}") if not log_file else None
    print(f"{arg}: {value}", end=' | ')
logging.info('--'*50) if not log_file else None
print('\n')

print(f'🟢 Logging info in "{filename_log}"')

""" load datasets """

if args.dataset_name.lower() == 'h2o':
    annotation_components = ['cam_pose', 'hand_pose', 'hand_pose_mano', 'obj_pose', 'obj_pose_rt', 'action_label', 'verb_label']
    my_preprocessor = MyPreprocessor('../mano_v1_2/models/', 'datasets/objects/mesh_1000/', args.root)
    h2o_data_dir = os.path.join(args.root, 'shards')
    train_input_tar_lists, train_annotation_tar_files = load_tar_split(h2o_data_dir, 'train')    
    val_input_tar_lists, val_annotation_tar_files = load_tar_split(h2o_data_dir, 'val')   
    num_classes = 4
    graph_input = 'coords'
else: # i.e. HO3D, POV-Surgery
    print(f'Loading training data ...', end=' ')
    # trainloader = [] # DEBUG
    trainloader = create_loader(args.dataset_name, args.root, 'train', batch_size=args.batch_size, num_kps3d=num_kps3d, num_verts=num_verts, other_params=other_params) # DEBUG
    print(f'✅ Training data loaded.')
    print(f'Loading validation data ...', end=' ')
    # DEBUG
    # valloader = trainloader # DEBUG
    valloader = create_loader(args.dataset_name, args.root, 'val', batch_size=args.batch_size, other_params=other_params)
    print(f'✅ Validation data loaded.')
    num_classes = 2 
    graph_input = 'heatmaps'

""" load model """
torch.cuda.empty_cache()

model = create_thor(num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_verts=num_verts, num_classes=num_classes, 
                                rpn_post_nms_top_n_train=num_classes-1, 
                                device=device, num_features=args.num_features, hid_size=args.hid_size,
                                photometric=args.photometric, graph_input=graph_input, dataset_name=args.dataset_name, testing=args.testing,
                                hands_connectivity_type=args.hands_connectivity_type,
                                multiframe=args.multiframe)

# pytorch_total_params = sum(p.numel() for p in model.parameters()) # DEBUG
# print(f'# params THOR-Net: {pytorch_total_params}') # DEBUG

print('THOR-Net is loaded')

if torch.cuda.is_available():
    model = model.cuda(args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

""" load saved model"""

if args.pretrained_model != '':
    print(f'🟢 Loading checkpoint "{args.pretrained_model.split(os.sep)[-2]}{os.sep}{args.pretrained_model.split(os.sep)[-1]}" ...')
    state_dict = torch.load(args.pretrained_model, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '')] = state_dict.pop(key)
        model.load_state_dict(state_dict)
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
    print(f'🟢 Model checkpoint "{args.pretrained_model.split(os.sep)[-2]}{os.sep}{args.pretrained_model.split(os.sep)[-1]}" loaded')
else:
    losses = []
    start = 0

"""define optimizer"""

criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d', 'palm']

""" training """

print('🟢 Begin training the network')

min_total_loss = float('inf')

for epoch in range(start, start + args.num_iterations):  # loop over the dataset multiple times
    
    train_loss2d = 0.0
    running_loss2d = 0.0
    running_loss3d = 0.0
    running_mesh_loss3d = 0.0
    running_photometric_loss = 0.0
    
    if 'h2o' in args.dataset_name.lower():
        h2o_info = (train_input_tar_lists, train_annotation_tar_files, annotation_components, args.buffer_size, my_preprocessor)
        trainloader = create_loader(args.dataset_name, h2o_data_dir, 'train', args.batch_size, h2o_info=h2o_info)

    pbar = tqdm(desc=f'Epoch {epoch+1} - train: ', total=len(trainloader))
    for i, tr_data in enumerate(trainloader):
        
        # get the inputs
        data_dict = tr_data
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
        inputs = {
            'inputs': [t['inputs'].to(device) for t in data_dict],
            'prev_frames': [t['prev_frames'] for t in data_dict if 'prev_frames' in t]
        }
        loss_dict, result = model(inputs, targets)
        
        # Calculate Loss
        loss = sum(loss_dict.values())
        
        # Backpropagate
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss2d += loss_dict['loss_keypoint'].data
        running_loss2d += loss_dict['loss_keypoint'].data
        running_loss3d += loss_dict['loss_keypoint3d'].data
        running_mesh_loss3d += loss_dict['loss_mesh3d'].data
        if 'loss_photometric' in loss_dict.keys():
            running_photometric_loss += loss_dict['loss_photometric'].data

        if (i+1) % args.log_batch == 0:    # print every args.log_iter mini-batches
            logging.info('[Epoch %d/%d, Processed data %d/%d] loss 2d: %.4f, loss 3d: %.4f, mesh loss 3d: %.4f, photometric loss: %.4f' % 
            (epoch + 1, start+args.num_iterations, i + 1, len(trainloader), running_loss2d / args.log_batch, running_loss3d / args.log_batch, 
            running_mesh_loss3d / args.log_batch, running_photometric_loss / args.log_batch))
            running_mesh_loss3d = 0.0
            running_loss2d = 0.0
            running_loss3d = 0.0
            running_photometric_loss = 0.0
            
        pbar.update(1)
    pbar.close()
    
    losses.append((train_loss2d / (i+1)).cpu().numpy())
    
    if (epoch+1) % args.snapshot_epoch == 0 and loss.data < min_total_loss: # save model only if total loss is lower than minimum reached
        torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
        np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))
        print(f'Model checkpoint (epoch {epoch+1}) saved in "{output_folder}"')
        # delete files from older epochs
        if epoch+1 > 1:
            files_to_delete = [x for x in os.listdir(output_folder) if f'model-' in x and f'model-{epoch+1}' not in x]
            for file in files_to_delete:
                try:
                    os.remove(os.path.join(output_folder, file))
                except:
                    pass
        min_total_loss = loss.data
        
    # ''' ------------------------ VALIDATION ------------------------ '''
    
    if (epoch+1) % args.val_epoch == 0:
        val_loss2d = 0.0
        val_loss3d = 0.0
        val_mesh_loss3d = 0.0
        val_photometric_loss = 0.0
        
        # model.module.transform.training = False
        
        if 'h2o' in args.dataset_name.lower():
            h2o_info = (val_input_tar_lists, val_annotation_tar_files, annotation_components, args.buffer_size, my_preprocessor)
            valloader = create_loader(args.dataset_name, h2o_data_dir, 'val', args.batch_size, h2o_info)

        pbar = tqdm(desc=f'Epoch {epoch+1} - val: ', total=len(valloader))
        for v, val_data in enumerate(valloader):
            
            # get the inputs
            data_dict = val_data
        
            # wrap them in Variable
            targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
            inputs = [t['inputs'].to(device) for t in data_dict]    
            loss_dict, result = model(inputs, targets)
            
            val_loss2d += loss_dict['loss_keypoint'].data
            val_loss3d += loss_dict['loss_keypoint3d'].data
            val_mesh_loss3d += loss_dict['loss_mesh3d'].data
            if 'loss_photometric' in loss_dict.keys():
                running_photometric_loss += loss_dict['loss_photometric'].data
                
            # visualizations
            '''if args.visualize: 
                from test_THOR import visualize2d
                path = data_dict[0]['path'].split(os.sep)[-1]
                if args.dataset_name=='ho3d' or args.dataset_name=='TEST_DATASET': # choose specific sequence to evaluate
                    if args.seq not in data_dict[0]['path']:
                        continue
                    if '_' in path:
                        path = path.split('_')[-1]
                    frame_num = int(path.split('.')[0])
                elif args.dataset_name=='povsurgery':
                    seq_name = data_dict[0]['path'].split(os.sep)[-2]
                else:
                    pass
                
                outputs = (result, loss_dict)
                
                predictions, img, palm, labels = prepare_data_for_evaluation(data_dict, outputs, img, keys, device, args.split)
                
                name = path.split(os.sep)[-1]
                output_dir = os.path.join(args.output_results, seq_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if (num_classes == 2 and 1 in predictions['labels']) or (num_classes == 4 and set([1, 2, 3]).issubset(predictions['labels'])):
                    visualize2d(img, predictions, labels, filename=f'{os.path.join(output_dir, name)}', palm=palm, evaluate=True)
                else:
                    cv2.imwrite(f'{os.path.join(output_dir, name)}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            '''
            pbar.update(1)
        pbar.close()
        
        # model.module.transform.training = True
        
        logging.info('Epoch %d/%d - val loss 2d: %.4f, val loss 3d: %.4f, val mesh loss 3d: %.4f, val photometric loss: %.4f' % 
                    (epoch + 1, start+args.num_iterations, val_loss2d / (v+1), val_loss3d / (v+1), val_mesh_loss3d / (v+1), running_photometric_loss / (v+1)))  
        print('Epoch %d/%d - val loss 2d: %.4f, val loss 3d: %.4f, val mesh loss 3d: %.4f, val photometric loss: %.4f' % 
                    (epoch + 1, start+args.num_iterations, val_loss2d / (v+1), val_loss3d / (v+1), val_mesh_loss3d / (v+1), running_photometric_loss / (v+1)))  
    
    if args.freeze and epoch == 0:
        logging.info('Freezing Keypoint RCNN ..')            
        freeze_component(model.module.backbone)
        freeze_component(model.module.rpn)
        freeze_component(model.module.roi_heads)

    # Decay Learning Rate
    scheduler.step()

logging.info('Finished Training')