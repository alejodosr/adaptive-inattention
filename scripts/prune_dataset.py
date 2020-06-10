#!/usr/bin/python3
"""Script for pruning the ILSVRC2015 - 2017 dataset in order to reduce the number of classes
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
import shutil

dirs = ['ILSVRC2015_VID_train_0000/',
        'ILSVRC2015_VID_train_0001/',
        'ILSVRC2015_VID_train_0002/',
        'ILSVRC2015_VID_train_0003/']

dataset_name = "aggregated_ILSVRC"
dirs_val = ['/home/alejo/Downloads/' + dataset_name + '/Data/VID/val/']
dirs_test = ['/home/alejo/Downloads/' + dataset_name + '/Data/VID/test/']

""" Whole set of classes """
classes_names = ['__background__',  # always index 0
                       'airplane', 'bicycle',
                       'bird', 'car',
                       'fox',
                       'lion',
                       'monkey']
classes_map = ['__background__',  # always index 0
                     'n02691156', 'n02834778',
                     'n01503061', 'n02958343',
                     'n02118333',
                     'n02129165',
                     'n02484322']

""" New reduced set of classes"""
# classes_names = ['__background__',  # always index 0
#                  'airplane',
#                  'motorcycle']
#
# classes_map = ['__background__',  # always index 0
#                'n02691156',
#                'n03790512']

for dir in dirs:
    seqs = np.sort(os.listdir(os.path.join('/home/alejo/Downloads/' + dataset_name + '/Data/VID/train/' + dir)))
    for seq in seqs:
        seq_path = os.path.join('/home/alejo/Downloads/' + dataset_name + '/Data/VID/train/', dir, seq)
        snippet_file = os.path.join('/home/alejo/Downloads/' + dataset_name + '/Data/VID/snippets/train/', dir, seq + ".mp4")
        relative_path = dir + seq
        image_list = np.sort(os.listdir(seq_path))
        count = 0
        for image in image_list:
            image_id = image.split('.')[0]
            anno_file = image_id + '.xml'
            anno_dir = os.path.join('/home/alejo/Downloads/' + dataset_name + '/Annotations/VID/train/', dir, seq)
            anno_path = os.path.join('/home/alejo/Downloads/' + dataset_name + '/Annotations/VID/train/', dir, seq, anno_file)
            objects = ET.parse(anno_path).findall("object")
            num_objs = len(objects)
            if num_objs == 0:  # discarding images without object
                continue
            else:
                # Find classes
                root = ET.parse(anno_path).getroot()
                for obj in root.findall('object'):
                    wanted_class = False
                    for label in classes_map:
                        if obj.find('name').text == label:
                            wanted_class = True

                if not wanted_class:
                    print(seq_path)
                    print(anno_dir)
                    print(snippet_file)
                    # Removing folders
                    try:
                        shutil.rmtree(seq_path)
                        print("Removing folder: " + seq_path)
                        try:
                            shutil.rmtree(anno_dir)
                            print("Removing folder: " + anno_dir)
                            try:
                                os.remove(snippet_file)
                                print("Removing folder: " + snippet_file)
                                break
                            except OSError as e:
                                print("Error: %s : %s" % (snippet_file, e.strerror))
                        except OSError as e:
                            print("Error: %s : %s" % (anno_dir, e.strerror))
                    except OSError as e:
                        print("Error: %s : %s" % (seq_path, e.strerror))

                count = count + 1
                print(relative_path + '/' + image_id)

for dir in dirs_val:
    seqs = np.sort(os.listdir(dir))
    for seq in seqs:
        seq_path = os.path.join(dir, seq)
        snippet_file = os.path.join('/home/alejo/Downloads/' + dataset_name + '/Data/VID/snippets/val/', seq + ".mp4")
        relative_path = dir + seq
        image_list = np.sort(os.listdir(seq_path))
        count = 0
        for image in image_list:
            image_id = image.split('.')[0]
            anno_file = image_id + '.xml'
            anno_dir = os.path.join('/home/alejo/Downloads/' + dataset_name + '/Annotations/VID/val/', seq)
            anno_path = os.path.join('/home/alejo/Downloads/' + dataset_name + '/Annotations/VID/val/', seq, anno_file)
            objects = ET.parse(anno_path).findall("object")
            num_objs = len(objects)
            if num_objs == 0:  # discarding images without object
                continue
            else:
                # Find classes
                root = ET.parse(anno_path).getroot()
                for obj in root.findall('object'):
                    wanted_class = False
                    for label in classes_map:
                        if obj.find('name').text == label:
                            wanted_class = True

                if not wanted_class:
                    print(seq_path)
                    print(anno_dir)
                    print(snippet_file)
                    # Removing folders
                    try:
                        shutil.rmtree(seq_path)
                        print("Removing folder: " + seq_path)
                        try:
                            shutil.rmtree(anno_dir)
                            print("Removing folder: " + anno_dir)
                            try:
                                os.remove(snippet_file)
                                print("Removing folder: " + snippet_file)
                                break
                            except OSError as e:
                                print("Error: %s : %s" % (snippet_file, e.strerror))
                        except OSError as e:
                            print("Error: %s : %s" % (anno_dir, e.strerror))
                    except OSError as e:
                        print("Error: %s : %s" % (seq_path, e.strerror))

                count = count + 1
                print(relative_path + '/' + image_id)
