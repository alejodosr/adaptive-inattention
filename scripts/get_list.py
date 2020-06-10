#!/usr/bin/python3
"""Script for creating text file containing sequences of all the video frames. Here we neglect all the frames where 
there is no object in it as it was done in the official implementation in tensorflow.
Global Variables
----------------
dirs : containing list of all the training dataset folders
dirs_val : containing path to val folder of dataset
dirs_test : containing path to test folder of dataset
"""
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os

# Parameter
dataset_dir = '/path/to/dataset'


dirs = ['ILSVRC2015_VID_train_0000/',
        'ILSVRC2015_VID_train_0001/',
        'ILSVRC2015_VID_train_0002/',
        'ILSVRC2015_VID_train_0003/']
dirs_val = [os.path.join(dataset_dir, 'Data/VID/val/')]
dirs_test = [os.path.join(dataset_dir, 'Data/VID/test/')]

get_shape_stats = False
if get_shape_stats:
    base_shape = (0, 0, 0)
    data_dict = {}

file_write_obj = open('../checkpoints/train_VID_list.txt', 'w')
for dir in dirs:
    seqs = np.sort(os.listdir(os.path.join(os.path.join(dataset_dir, 'Data/VID/train/') + dir)))
    for seq in seqs:
        seq_path = os.path.join(os.path.join(dataset_dir, 'Data/VID/train/'), dir, seq)
        relative_path = dir + seq
        image_list = np.sort(os.listdir(seq_path))
        count = 0
        for image in image_list:
            if get_shape_stats:
                # Count number of images an resolutions
                # print(dataset_dir + "/" + 'Data/VID/train/' + relative_path + "/" + image)
                img = cv2.imread(dataset_dir + "/" + 'Data/VID/train/' + relative_path + "/" + image)
                if img.shape != base_shape:
                    print(img.shape)
                    if str(img.shape) not in data_dict:
                        data_dict.update({str(img.shape): 0})
                    base_shape = img.shape
                data_dict[str(img.shape)] += 1

            image_id = image.split('.')[0]
            anno_file = image_id + '.xml'
            anno_path = os.path.join(os.path.join(dataset_dir, 'Annotations/VID/train/'), dir, seq, anno_file)
            objects = ET.parse(anno_path).findall("object")
            num_objs = len(objects)
            if num_objs == 0:  # discarding images without object
                continue
            else:
                count = count + 1
                file_write_obj.writelines(relative_path + '/' + image_id)
                file_write_obj.write('\n')
file_write_obj.close()
if get_shape_stats:
    # Print statistics of resolutions
    print(data_dict)

file_write_obj = open('../checkpoints/val_VID_list.txt', 'w')
for dir in dirs_val:
    seqs = np.sort(os.listdir(dir))
    for seq in seqs:
        seq_path = os.path.join(dir, seq)
        image_list = np.sort(os.listdir(seq_path))
        count = 0
        for image in image_list:
            image_id = image.split('.')[0]
            anno_file = image_id + '.xml'
            anno_path = os.path.join(os.path.join(dataset_dir, 'Annotations/VID/val/'), seq, anno_file)
            objects = ET.parse(anno_path).findall("object")
            num_objs = len(objects)
            if num_objs == 0:
                continue
            else:
                count = count + 1
                if count <= 20:
                    file_write_obj.writelines(seq + '/' + image_id)
                    file_write_obj.write('\n')

file_write_obj.close()
# file_write_obj = open('test_VID_list.txt','w')
# for dir in dirs_test:
# 	seqs = np.sort(os.listdir(dir))
# 	for seq in seqs:
# 		seq_path = os.path.join(dir,seq)
# 		image_list = np.sort(os.listdir(seq_path))
# 		for image in image_list:
# 			file_write_obj.writelines(seq+image)
# 			file_write_obj.write('\n')

# file_write_obj.close()
