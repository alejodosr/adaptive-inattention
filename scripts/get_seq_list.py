#!/usr/bin/python3
"""Script for creating text file containing sequences of 10 frames of particular video. Here we neglect all the frames where 
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

seq_len = 10


dirs = ['ILSVRC2015_VID_train_0000/',
        'ILSVRC2015_VID_train_0001/',
        'ILSVRC2015_VID_train_0002/',
        'ILSVRC2015_VID_train_0003/']
dirs_val = [os.path.join(dataset_dir, 'Data/VID/val/')]
dirs_test = [os.path.join(dataset_dir, 'Data/VID/test/')]

file_write_obj = open('../checkpoints/train_VID_seqs_list.txt', 'w')
for dir in dirs:
    seqs = np.sort(os.listdir(os.path.join(os.path.join(dataset_dir, 'Data/VID/train/') + dir)))
    for seq in seqs:
        seq_path = os.path.join(os.path.join(dataset_dir, 'Data/VID/train/'), dir, seq)
        relative_path = dir + seq
        image_list = np.sort(os.listdir(seq_path))
        count = 0
        filtered_image_list = []
        for image in image_list:
            image_id = image.split('.')[0]
            anno_file = image_id + '.xml'
            anno_path = os.path.join(os.path.join(dataset_dir, 'Annotations/VID/train/'), dir, seq, anno_file)
            objects = ET.parse(anno_path).findall("object")
            num_objs = len(objects)
            if num_objs == 0:  # discarding images without object
                continue
            else:
                count = count + 1
                filtered_image_list.append(relative_path + '/' + image_id)
        for i in range(0, int(count / seq_len)):
            seqs = ''
            for j in range(0, seq_len):
                seqs = seqs + filtered_image_list[seq_len * i + j] + ','
            seqs = seqs[:-1]
            file_write_obj.writelines(seqs)
            file_write_obj.write('\n')
file_write_obj.close()
file_write_obj = open('../checkpoints/val_VID_seqs_list.txt', 'w')
seq_list = []
with open('../checkpoints/val_VID_list.txt') as f:
    for line in f:
        seq_list.append(line.rstrip())
for i in range(0, int(len(seq_list) / seq_len)):
    # image_path = seq_list[seq_len*i].split('/')[0]
    # seqs = image_path+'/'+':'
    seqs = ''
    for j in range(0, seq_len):
        seqs = seqs + seq_list[seq_len * i + j] + ','
    seqs = seqs[:-1]
    file_write_obj.writelines(seqs)
    file_write_obj.write('\n')
file_write_obj.close()
file_write_obj = open('../checkpoints/test_VID_seqs_list.txt', 'w')
for dir in dirs_test:
    seqs = np.sort(os.listdir(dir))
    for seq in seqs:
        seq_path = os.path.join(dir, seq)
        image_list = np.sort(os.listdir(seq_path))
        for image in image_list:
            file_write_obj.writelines(seq + image)
            file_write_obj.write('\n')

file_write_obj.close()
