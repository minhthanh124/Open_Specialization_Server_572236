root_dir = '/home/minhthanh/Open_Specialization_572236/Server/TransPose'

import torch
import torch.onnx
import sys
import glob
from pathlib import Path
sys.path.append(root_dir + '/lib')
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from core.evaluate import accuracy
import models
import torch.nn as nn
import os
import socket
import json
import struct
import random
import shutil
from torch.utils.data import Dataset
from PIL import Image
import dataset
import time
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

import threading
import torch.optim as optim
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary


onnx_file_path = root_dir + '/pretrained_model_onnx/transpose_pretrained_model.onnx'
source_folder = root_dir + '/custom_scripts/'
destination_folder_1 = root_dir + '/data/coco/images/train2017/'
destination_folder_2 = root_dir +'/data/coco/images/val2017/'
annotation_json_trainPath = root_dir + '/data/coco/annotations/person_keypoints_train2017.json'
annotation_json_valPath = root_dir + '/data/coco/annotations/person_keypoints_val2017.json'
folder_path_json = root_dir + '/data/coco/annotations/'
final_output_dir = root_dir + '/out/'
root_label_train = root_dir + '/data_temp/annotations/old/tmp/person_keypoints_train2017.json'
root_label_validate = root_dir + '/data_temp/annotations/old/tmp/person_keypoints_val2017.json'
annotation_dir = root_dir + '/data/coco/annotations/'
config_file = root_dir + '/experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml'
pretrained_weights = final_output_dir + 'results_transpose_r_a4_car/final_state.pth'


def removeUnsuedFiles():
    jpg_files_train = glob.glob(os.path.join(destination_folder_1, '*.jpg'))
    jpg_files_test = glob.glob(os.path.join(destination_folder_2, '*.jpg'))
    json_files = glob.glob(os.path.join(folder_path_json, '*.json'))
    files_to_delete = jpg_files_train + jpg_files_test + json_files
    for file_path in files_to_delete:
    	if os.path.exists(file_path):
           try:
              os.remove(file_path)
              print(f"Deleted: {file_path}")
           except Exception as e:
              print(f"Error deleting {file_path}: {e}")
           
    shutil.copy(root_label_train, annotation_dir)
    shutil.copy(root_label_validate, annotation_dir)

def send_file(filename, conn):
    with open(filename, 'rb') as file:
        data = file.read()
        conn.sendall(data)

def sendModel(client_socket):
    send_file(onnx_file_path, client_socket)
    client_socket.close()

def save_image(image_data, filename):
    file_path = os.path.join(source_folder, filename)
    with open(file_path, "wb") as f:
        f.write(image_data)

def save_json(json_data, filename="person_keypoints_train2017.json"):
    file_path = os.path.join(source_folder, filename)
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)

def handle_request(client_sockets, addr):
    i = 0
    i = i + 1
    while True:
        try:
            # Read message length
            msg_len_data = client_sockets.recv(4)
            if len(msg_len_data) < 4:
               client_sockets.close()
               continue
            
            msg_len = struct.unpack('>I', msg_len_data)[0]

            # Read the complete message

            data = b''
            while len(data) < msg_len:
                packet = client_sockets.recv(msg_len - len(data))
                if not packet:
                    break
                data += packet
            
            if len(data) != msg_len:
                print("Received data length does not match the expected length.")
                client_sockets.close()
                continue

            request_len = struct.unpack('>I', data[:4])[0]
            if b'\xD9' in data[:4]:
                print('Receiving Train Request! Starting training model ... ')
                startProcess()
                print('New model is now available!')
                with open(onnx_file_path, 'rb') as file:
                     chunk = file.read(8192)
                     while chunk:
                           client_sockets.send(chunk)
                           chunk = file.read(8192)
                print('Send onnx to client sucessfully!')
                continue

            json_len = struct.unpack('>I', data[4:8])[0]
            json_data = json.loads(data[8:8 + json_len])
            save_json(json_data, f"jsonfile_0{i}.json")
            image_data = data[12 + json_len:]  # Skipping JSON length (4 bytes) and JSON data

            save_json(json_data, f"jsonfile_0{i}.json")
            with open(os.path.join(source_folder, f"jsonfile_0{i}.json"), 'r') as file_json:
                data = json.load(file_json)
            name = data['images'][0]['file_name']
            save_image(image_data, f"{name}")
            os.rename(source_folder + f"jsonfile_0{i}.json", source_folder + name.split(".jpg")[0] +".json")
            print("Data received and saved")
        
        except Exception as e:
            print(f"Error processing the data: {e}")

    client_sockets.close()

def start_server():
    name = ""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 7000))
    server_socket.listen(5)
    print("Server is listening on port 7000")

    while True:
        client_sockets, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_thread = threading.Thread(target=handle_request, args=(client_sockets, addr))
        client_thread.start()
        print(f"Started thread for {addr}")

def moveFile():
    files_list = []
    file_list_annotation = []
    file_tmp = []

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            #all 
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                files_list.append(os.path.join(root, file))
            elif file.endswith(".json"):
                file_list_annotation.append(os.path.join(root, file))

    file_count = len(files_list)
    print(file_count)

    # print files_list
    random.shuffle(files_list)
    filesToCopy = random.sample(files_list, int(file_count*0.7))  #prints two random files from list 


    # if destination dir does not exists, create it
    if os.path.isdir(destination_folder_1) == False:
            os.makedirs(destination_folder_1)

    # iteraate over all random files and move them
    for file in filesToCopy:
        shutil.move(file, destination_folder_1)

    for root, dirs, files in os.walk(destination_folder_1):
        for file in files:
            if file.endswith(".jpg"):
                file_tmp.append(file.split(".jpg")[0])

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".json"):
                if file.split(".json")[0] in file_tmp:
                    shutil.move(source_folder + file, destination_folder_1)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".json"):
                shutil.move(source_folder + file, destination_folder_2)

def jsonGenetate(pathToFolder, pathToJson):
    for root, dirs, files in os.walk(pathToFolder):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                # Load the JSON data from the file
                with open(json_file_path, 'r') as file_json:
                    data = json.load(file_json)
                new_subnodes_image = {
                    "license": data['images'][0]["license"],
                    "file_name": data['images'][0]["file_name"],
                    "coco_url": data['images'][0]["coco_url"],
                    "width": data['images'][0]["width"],
                    "height": data['images'][0]["height"],
                    "date_captured": data['images'][0]["date_captured"],
                    "flickr_url": data['images'][0]["flickr_url"],
                    "id": data['images'][0]["id"]
                }

                new_subnodes_anno = {
                    "segmentation": data['annotations'][0]["segmentation"],
                    "num_keypoints": data['annotations'][0]["num_keypoints"],
                    "area": data['annotations'][0]["area"],
                    "iscrowd": data['annotations'][0]["iscrowd"],
                    "keypoints": data['annotations'][0]["keypoints"],
                    "image_id": data['annotations'][0]["image_id"],
                    "bbox": data['annotations'][0]["bbox"],
                    "category_id": data['annotations'][0]["category_id"],
                    "id": data['annotations'][0]["id"],
            }

                anno_open_file = os.path.join(root, pathToJson)
                with open(anno_open_file, 'r+') as file_dirs:
                    datax = json.load(file_dirs)
                    datax["images"].append(new_subnodes_image)
                    datax["annotations"].append(new_subnodes_anno)
                    file_dirs.seek(0)
                    json.dump(datax, file_dirs, indent = 4)

def removeFile(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                os.remove(path + file)

def startProcess():
    removeUnsuedFiles() 
    moveFile()
    jsonGenetate(destination_folder_1, annotation_json_trainPath)
    jsonGenetate(destination_folder_2, annotation_json_valPath)
    removeFile(destination_folder_1)
    removeFile(destination_folder_2)
    runModel()

# Model -------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        
def create_meta_tasks(cfg, normalize):
    tasks = []
    for dirpath, dirnames, filenames in os.walk(root_dir + '/data/coco/'):
        for sub in dirnames:
        	if 'object_' in sub:
		        train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
		                cfg, cfg.DATASET.ROOT, sub + '/' + cfg.DATASET.TRAIN_SET, True,
		                transforms.Compose([
		                    transforms.ToTensor(),
		                    normalize,
		                ])
		            )

		        train_loader = torch.utils.data.DataLoader(
		                train_dataset,
		                batch_size=1,
		                shuffle=cfg.TRAIN.SHUFFLE,
		            )

		        valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
		                cfg, cfg.DATASET.ROOT, sub + '/' + cfg.DATASET.TEST_SET, False,
		                transforms.Compose([
		                    transforms.ToTensor(),
		                    normalize,

		                 ])
		             )

		        valid_loader = torch.utils.data.DataLoader(
		                valid_dataset,
		                batch_size=1,
		                shuffle=False,
		             )

		        task = {
		            'train_loader': train_loader,
		            'valid_loader': valid_loader
		        }
		        tasks.append(task)

    return tasks
    		
def runMetaTraining(tasks, model):
    num_meta_epochs = 100  # Number of epochs for meta-training
    num_fine_tune_epochs = 10  # Number of epochs for fine-tuning
    meta_batch_size = 2  # Number of tasks in each meta-batch
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    )
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    fine_tune_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    acc = AverageMeter()

    for meta_epoch in range(num_meta_epochs):
        # Sample a batch of tasks (e.g., different object categories)
        meta_tasks = tasks

        meta_loss = torch.tensor(0.0, requires_grad=True).to(device)

        for task in meta_tasks:
            # Initialize task-specific data loader
            task_train_loader, task_valid_loader = task['train_loader'], task['valid_loader']

            # Inner loop: train on the current task (object category)
            task_loss = torch.tensor(0.0).to(device)
            for input, target, target_weight, meta in task_train_loader:
                input, target, target_weight = input.to(device), target.to(device), target_weight.to(device)

                outputs = model(input)
                loss = criterion(outputs, target, target_weight)

                fine_tune_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                fine_tune_optimizer.step()
                task_loss += loss

                _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
                acc.update(avg_acc, cnt)
                msg = 'Epoch: [{0}]\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          meta_epoch, acc=acc)
                print(msg)

            # Aggregate task loss for meta-update
            meta_loss = meta_loss + (task_loss / len(task_train_loader))

        # Meta-update: Adjust the model parameters across tasks
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        print(f"Meta Epoch [{meta_epoch}/{num_meta_epochs}], Meta Loss: {meta_loss/meta_batch_size:.5f}")

    # Save the meta-trained model
    torch.save(model.state_dict(), final_output_dir + 'meta_trained_transpose_h-a4.pth')

def runModel():

    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    model = models.transpose_r.get_pose_net(cfg, is_train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(pretrained_weights, map_location='cpu'))

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # tasks = create_meta_tasks(cfg, normalize)
    # runMetaTraining(tasks, model)


    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=cfg.TRAIN.SHUFFLE,
        )

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
             ])
         )

    valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
         )

    tb_log_dir = ''
    # Define the loss function and optimizer
    criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    )
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
            final_output_dir, 'checkpoint.pth'
    )
    num_epochs = 200
    running_loss = 0.0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)
    # Training loop

    model.train()
    
    epoch_time = 0
    end = 0
    time_start = time.time()
    for epoch in range(num_epochs):  # Adjust the number of epochs
        cal_time = time.time() - end
        print('Time processing at epoch {0} is: {1}'.format(epoch-1, cal_time))
        end = time.time()
        print('Total Time processing at epoch {0} is: {1}'.format(epoch-1, end - time_start))
        
        for i, (input, target, target_weight, meta) in enumerate(train_loader):
            data_time.update(time.time() - end)
            outputs = model(input)
            target = target.to(device)
            target_weight = target_weight.to(device)
            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure accuracy and record loss
            
            losses.update(loss.item(), input.size(0))

            _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)

            #if i % cfg.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch, i, len(train_loader), loss=losses, batch_time=batch_time, speed=input.size(0)/batch_time.val, data_time=data_time, acc=acc)
            print(msg)

        perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir
        )

        lr_scheduler.step()

        if perf_indicator >= best_perf:
           best_perf = perf_indicator
           best_model = True
        else:
            best_model = False
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    
    time_end = time.time()
    total_time = time_end - time_start
    print('Total training time is: {0}\t'.format(total_time))
    final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
    )
    torch.save(model.state_dict(), final_model_state_file)

    #----------------------------- Convert ONNX -------------------------------

    # Define dummy input (adjust the size if needed)
    dummy_input = torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])  # H x W

    # Export the model to ONNX

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"Model successfully exported to {onnx_file_path}")

start_server()