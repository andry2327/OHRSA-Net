import re
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

plt.rcParams['figure.dpi'] = 300
FONTSIZE = 18

'''
def parse_logs(log_file):
    
    epochs_train, val_epochs = [], []
    losses_2d = []
    losses_3d = []
    mesh_losses_3d = []
    photometric_losses = []
    val_losses_2d = []
    val_losses_3d = []
    val_mesh_losses_3d = []
    val_photometric_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            
            if '--' not in line and 'Epoch' in line and '[' in line and ']' in line: # train loss
                line_splitted = line.split(' ')
                epoch = int(line_splitted[1].split('/')[0])
                loss_2d = float(line_splitted[7].strip(','))
                loss_3d = float(line_splitted[10].strip(','))
                mesh_loss_3d = float(line_splitted[14].strip(','))
                photometric = float(line_splitted[17].strip('\n'))
                epochs_train.append(epoch)
                losses_2d.append(loss_2d)
                losses_3d.append(loss_3d)
                mesh_losses_3d.append(mesh_loss_3d)
                photometric_losses.append(photometric)
            elif '--' not in line and 'Epoch' in line and '[' not in line and ']' not in line: # val loss
                line_splitted = line.split(' ')
                epoch = int(line_splitted[1].split('/')[0])
                loss_2d = float(line_splitted[6].strip(','))
                loss_3d = float(line_splitted[10].strip(','))
                mesh_loss_3d = float(line_splitted[15].strip(','))
                photometric = float(line_splitted[19].strip('\n'))
                val_epochs.append(epoch)
                val_losses_2d.append(loss_2d)
                val_losses_3d.append(loss_3d)
                val_mesh_losses_3d.append(mesh_loss_3d)
                val_photometric_losses.append(photometric)
                
    return epochs_train, val_epochs, losses_2d, losses_3d, mesh_losses_3d, photometric_losses, val_losses_2d, val_losses_3d, val_mesh_losses_3d, val_photometric_losses''''''
'''
def parse_logs(log_file):
    epochs_train = {}
    val_epochs = {}
    losses_2d = {}
    losses_3d = {}
    mesh_losses_3d = {}
    photometric_losses = {}
    val_losses_2d = {}
    val_losses_3d = {}
    val_mesh_losses_3d = {}
    val_photometric_losses = {}

    with open(log_file, 'r') as f:
        for line in f:
            if '--' not in line and 'Epoch' in line:
                if '[' in line and ']' in line:  # train loss
                    line_splitted = line.split(' ')
                    epoch = int(line_splitted[1].split('/')[0])
                    loss_2d = float(line_splitted[7].strip(','))
                    loss_3d = float(line_splitted[10].strip(','))
                    mesh_loss_3d = float(line_splitted[14].strip(','))
                    photometric = float(line_splitted[17].strip('\n'))
                    epochs_train[epoch] = epoch
                    if epoch not in losses_2d.keys():
                        losses_2d[epoch] = []
                    losses_2d[epoch].append(loss_2d)
                    if epoch not in losses_3d.keys():
                        losses_3d[epoch] = []
                    losses_3d[epoch].append(loss_3d)
                    if epoch not in mesh_losses_3d.keys():
                        mesh_losses_3d[epoch] = []
                    mesh_losses_3d[epoch].append(mesh_loss_3d)
                    if epoch not in photometric_losses.keys():
                        photometric_losses[epoch] = []
                    photometric_losses[epoch].append(photometric)
                elif '[' not in line and ']' not in line:  # val loss
                    line_splitted = line.split(' ')
                    epoch = int(line_splitted[1].split('/')[0])
                    loss_2d = float(line_splitted[6].strip(','))
                    loss_3d = float(line_splitted[10].strip(','))
                    mesh_loss_3d = float(line_splitted[15].strip(','))
                    photometric = float(line_splitted[19].strip('\n'))
                    val_epochs[epoch] = epoch
                    val_losses_2d[epoch] = loss_2d
                    val_losses_3d[epoch] = loss_3d
                    val_mesh_losses_3d[epoch] = mesh_loss_3d
                    val_photometric_losses[epoch] = photometric

    # Convert dictionaries to lists sorted by epoch
    epochs_train = sorted(epochs_train.values())
    val_epochs = sorted(val_epochs.values())
    losses_2d = [np.mean(losses_2d[epoch]) for epoch in epochs_train]
    losses_3d = [np.mean(losses_3d[epoch] )for epoch in epochs_train]
    mesh_losses_3d = [np.mean(mesh_losses_3d[epoch]) for epoch in epochs_train]
    photometric_losses = [np.mean(photometric_losses[epoch]) for epoch in epochs_train]
    val_losses_2d = [val_losses_2d[epoch] for epoch in val_epochs]
    val_losses_3d = [val_losses_3d[epoch] for epoch in val_epochs]
    val_mesh_losses_3d = [val_mesh_losses_3d[epoch] for epoch in val_epochs]
    val_photometric_losses = [val_photometric_losses[epoch] for epoch in val_epochs]

    return epochs_train, val_epochs, losses_2d, losses_3d, mesh_losses_3d, photometric_losses, val_losses_2d, val_losses_3d, val_mesh_losses_3d, val_photometric_losses

def plot_losses(log_file, out_path=''):
    epochs_train, val_epochs, losses_2d, losses_3d, mesh_losses_3d, photometric_losses, val_losses_2d, val_losses_3d, val_mesh_losses_3d, val_photometric_losses = parse_logs(log_file)
    
    fig, axs = plt.subplots(4, 2, figsize=(30, 40))

    # Define fontsize
    fontsize = FONTSIZE

    # Plot training loss 2D
    axs[0, 0].plot(epochs_train, losses_2d, label='Loss 2D', marker='o')
    axs[0, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[0, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[0, 0].set_title('Training Loss 2D', fontsize=fontsize)
    axs[0, 0].legend(fontsize=fontsize)
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[0, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot training loss 3D
    axs[1, 0].plot(epochs_train, losses_3d, label='Loss 3D', marker='o')
    axs[1, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[1, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[1, 0].set_title('Training Loss 3D', fontsize=fontsize)
    axs[1, 0].legend(fontsize=fontsize)
    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[1, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot training mesh loss 3D
    axs[2, 0].plot(epochs_train, mesh_losses_3d, label='Mesh Loss 3D', marker='o')
    axs[2, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[2, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[2, 0].set_title('Training Mesh Loss 3D', fontsize=fontsize)
    axs[2, 0].legend(fontsize=fontsize)
    axs[2, 0].grid(True)
    axs[2, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[2, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot training photometric loss
    axs[3, 0].plot(epochs_train, photometric_losses, label='Photometric Loss', marker='o')
    axs[3, 0].set_xlabel('Epochs', fontsize=fontsize)
    axs[3, 0].set_ylabel('Loss', fontsize=fontsize)
    axs[3, 0].set_title('Training Photometric Loss', fontsize=fontsize)
    axs[3, 0].legend(fontsize=fontsize)
    axs[3, 0].grid(True)
    axs[3, 0].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[3, 0].tick_params(axis='y', labelsize=fontsize)

    # Plot validation loss 2D
    axs[0, 1].plot(val_epochs, val_losses_2d, label='Validation Loss 2D', marker='x')
    axs[0, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[0, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[0, 1].set_title('Validation Loss 2D', fontsize=fontsize)
    axs[0, 1].legend(fontsize=fontsize)
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[0, 1].tick_params(axis='y', labelsize=fontsize)

    # Plot validation loss 3D
    axs[1, 1].plot(val_epochs, val_losses_3d, label='Validation Loss 3D', marker='x')
    axs[1, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[1, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[1, 1].set_title('Validation Loss 3D', fontsize=fontsize)
    axs[1, 1].legend(fontsize=fontsize)
    axs[1, 1].grid(True)
    axs[1, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[1, 1].tick_params(axis='y', labelsize=fontsize)

    # Plot validation mesh loss 3D
    axs[2, 1].plot(val_epochs, val_mesh_losses_3d, label='Validation Mesh Loss 3D', marker='x')
    axs[2, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[2, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[2, 1].set_title('Validation Mesh Loss 3D', fontsize=fontsize)
    axs[2, 1].legend(fontsize=fontsize)
    axs[2, 1].grid(True)
    axs[2, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[2, 1].tick_params(axis='y', labelsize=fontsize)

    # Plot validation photometric loss
    axs[3, 1].plot(val_epochs, val_photometric_losses, label='Validation Photometric Loss', marker='x')
    axs[3, 1].set_xlabel('Epochs', fontsize=fontsize)
    axs[3, 1].set_ylabel('Loss', fontsize=fontsize)
    axs[3, 1].set_title('Validation Photometric Loss', fontsize=fontsize)
    axs[3, 1].legend(fontsize=fontsize)
    axs[3, 1].grid(True)
    axs[3, 1].tick_params(axis='x', rotation=45, labelsize=fontsize)
    axs[3, 1].tick_params(axis='y', labelsize=fontsize)

    plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.95, bottom=0.05)
    
    file_name = 'loss_plots.png'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    path_out = os.path.join(out_path, file_name)
    print(f'Plots saved in "{path_out}"')
    plt.tight_layout()
    plt.savefig(path_out, dpi=300)
    #plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation losses from log file.')
    parser.add_argument('--log_file', type=str, help='Path to the log file')
    parser.add_argument('--output_path', required=False, type=str, help='Path where to save plots')
    args = parser.parse_args()
    
    plot_losses(args.log_file, args.output_path)
    # file = '/Users/andreapellegrino/Downloads/log_Training-100samples--20-06-2024_17-08.txt'
    # out_path = '/Users/andreapellegrino/Downloads'
    # plot_losses(file, out_path)

if __name__ == "__main__":
    main()