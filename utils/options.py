# -*- coding: utf-8 -*-
import argparse

def parse_args_function():
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.

    parser.add_argument(
        "--testing",
        action='store_true',
        help="Define if model training or testing"
    )
    
    parser.add_argument(
        "--dataset_name",
        default='ho3d', # or 'h2o'
        help="Name of the dataset"
    )

    parser.add_argument(
        "--mano_root",
        default='../mano_v1_2/models',
        help="Path to 'mano_v1_2' folder"
    )
    
    parser.add_argument(
        "--output_results",
        default='/content/drive/MyDrive/Thesis/THOR-Net_based_work',
        help="Path to save outputs results"
    )
    
    parser.add_argument(
        "--checkpoint_model",
        default='/content/THOR-Net/checkpoints/THOR-Net_trained_on_POV-Surgery_object_False/Training_sample--13-06-2024_15-22/model-1.pkl',
        help="Path to model to evaluate"
    )
    
    parser.add_argument(
        "--hands_connectivity_type",
        default='',
        help="Type of hand joints connectivity between simple, extended, full"
    )
    
    parser.add_argument(
        "--multiframe",
        action='store_true',
        help="Enable multiframe training"
    )
    
    parser.add_argument(
        "--bloodiness",
        action='store_true',
        help="Inject bloodiness feature"
    )

    parser.add_argument(
        "--obj_root",
        default='../datasets/objects/mesh_1000/book.obj',
        help="Path to 'book.obj' file"
    )

    parser.add_argument(
        "--root",
        default='./datasets/ho3d/',
        help="Input image, directory"
    )
    
    parser.add_argument(
        "--adj_matrix_root", 
        default='/datasets/adj_matrix', 
        help='Root folder for fixed adjacency matrix')
    
    parser.add_argument(
        "--output_file",
        default='./checkpoints/model-',
        help="Prefix of output pkl filename"
    )
    
    # Optional arguments.
    parser.add_argument(
        "--pretrained_model",
        default='',
        help="Load trained model weights file."
    )
    parser.add_argument(
        "--hdf5_path",
        default='',
        help="Path to HDF5 files to load to the memory for faster training, only suitable for sufficient memory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default = 512,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--gpu_number",
        type=int,
        nargs='+',
        default = [0],
        help="Identifies the GPU number to use."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Identifies the optimizer learning rate."
    )
    parser.add_argument(
        "--lr_step",
        type=int,
        default=100,
        help="Identifies the adaptive learning rate step size."
    )
    parser.add_argument(
        "--lr_step_gamma",
        type=float,
        default=0.9,
        help="Identifies the adaptive learning rate step gamma."
    )
    parser.add_argument(
        "--log_batch",
        type=int,
        default=100,
        help="Show log samples."
    )
    parser.add_argument(
        "--val_epoch",
        type=int,
        default=1,
        help="Run validation on epochs."
    )
    parser.add_argument(
        "--snapshot_epoch",
        type=int,
        default=1,
        help="Save snapshot epochs."
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Maximum number of epochs."
    )

    parser.add_argument(
        "--object",
        action='store_true',
        help="Generate 3D pose or mesh for the object"
    )

    parser.add_argument(
        "--num_features",
        type=int,
        default=2048,
        help="Number of features passed to coarse-to-fine network"
    )
    
    parser.add_argument(
        "--use_autocast",
        action='store_true',
        help="Enable mixed precision training"
    )
    
    parser.add_argument(
        "--keypoints2d_extractor_path",
        default='best.pt',
        help="Keypoints 2D extractor model path"
    )

    parser.add_argument(
        "--hid_size", 
        type=int, 
        default=128,
        help="hidden layer size in graformer"
    )

    parser.add_argument(
        "--freeze",
        action='store_true',
        help="Freeze RPN after first epoch"
    )

    # H2O specific

    parser.add_argument(
        "--buffer_size",
        type=int,
        default = 1000,
        help="Shuffle buffer size"
    )

    parser.add_argument(
        "--photometric",
        action='store_true',
        help="Generate textured mesh using photometric loss"
    )

    # For Testing
    parser.add_argument(
        "--seq", 
        default='MPM13', 
        help="Sequence Name"
    )

    parser.add_argument(
        "--checkpoint_folder", 
        default='ho',
        help="the folder of the pretrained model"
    )

    parser.add_argument(
        "--checkpoint_id", 
        type=int, 
        default=1,
        help="the id of the pretrained model"
    )

    parser.add_argument(
        "--visualize", 
        action='store_true', 
        help="Visualize results?"
    )

    parser.add_argument(
        "--split",
        default='test', 
        help="Which subset to evaluate on"
    )



    args = parser.parse_args()
    return args
