import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
from dataloaders.transforms import preloaded_rng
from copy import deepcopy

class YOLODatasetVID:

    def __init__(self, root, transform=None, target_transform=None, is_val=False,
                 label_file=None, seq_length=10, batch_size=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.seq_length = seq_length
        if is_val:
            self.subdir = 'val/img'
            self.image_sets_file = self.get_seq_files(os.path.join(self.root, self.subdir),
                                                      seq_length=self.seq_length, ext='.jpg')
            self.anno_sets_file = self.get_seq_files(os.path.join(self.root, self.subdir),
                                                     seq_length=self.seq_length, ext='.txt')
            self.shape_sets_file = self.get_seq_files(os.path.join(self.root, self.subdir),
                                                      seq_length=self.seq_length, ext='.shape')
        else:
            self.subdir = 'train/img'
            self.image_sets_file = self.get_seq_files(os.path.join(self.root, self.subdir),
                                                      seq_length=self.seq_length, ext='.jpg')
            self.anno_sets_file = self.get_seq_files(os.path.join(self.root, self.subdir),
                                                     seq_length=self.seq_length, ext='.txt')
            self.shape_sets_file = self.get_seq_files(os.path.join(self.root, self.subdir),
                                                      seq_length=self.seq_length, ext='.shape')

        rem = len(self.image_sets_file) % batch_size
        if rem != 0:
            self.image_sets_file = self.image_sets_file[:-rem]
            self.anno_sets_file = self.anno_sets_file[:-rem]
            self.shape_sets_file = self.shape_sets_file[:-rem]

        self.ids = []
        for seq_i in range(len(self.image_sets_file)):
            seq_ids = []
            for image_file in self.image_sets_file[seq_i]:
                seq_ids.append(image_file.split('/')[-1].split('.')[0])
            self.ids.append(seq_ids)

        # if the labels file exists, read in the class names
        label_file_name = self.root / "train/obj.names"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("YOLO Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND', 'dron')

        self._classes_names = self.class_names

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        images = []
        boxes_seq = []
        labels_seq = []
        # seed = round(time.time()*10000) #use current time as seed for whole sequence
        rng = preloaded_rng(1000)
        for i in range(self.seq_length):
            local_rng = deepcopy(rng)
            # torch.manual_seed(seed)
            image = self._read_image(index, i)
            boxes, labels = self._get_annotation(self.ids[index][i])
            if self.transform:
                image, boxes, labels = self.transform(image, boxes, labels, local_rng)
            if self.target_transform:
                boxes, labels = self.target_transform(boxes, labels)
            images.append(image)
            boxes_seq.append(boxes)
            labels_seq.append(labels)

        return images, boxes_seq, labels_seq

    def get_image(self, index, seq_pos):
        image = self._read_image(index, seq_pos)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index, seq_pos):
        image_id = self.ids[index][seq_pos]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.image_sets_file)

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(os.path.join(self.root, self.subdir), image_id + '.txt')
        with open(os.path.join(os.path.join(self.root, self.subdir), image_id + '.shape')) as fp:
            line = fp.readline().split(' ')
            shape = (int(line[0]), int(line[1]), int(line[2]))

        boxes = []
        labels = []
        with open(annotation_file) as fp:
            for cnt, line in enumerate(fp):
                line_ = line.split(' ')
                x1 = float(line_[1]) * float(shape[1])
                y1 = float(line_[2]) * float(shape[0])
                x2 = x1 + float(line_[3]) * float(shape[1])
                y2 = y1 + float(line_[4]) * float(shape[0])
                w = x2 - x1
                h = y2 - y1
                x1 -= w/2
                y1 -= h/2
                x2 -= w/2
                y2 -= h/2

                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[self._classes_names[int(line_[0]) + 1]])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, index, seq_pos):
        image_file = self.image_sets_file[index][seq_pos]
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_seq_files(self, dir, seq_length, ext='.jpg'):
        file_list = []
        for root, dirs, files in os.walk(dir):
            i = 0
            seq_list = []
            for file in sorted(files):
                if file.endswith(ext):
                    if i == 0:
                        initial_seq_id = file.split('_')[0]

                    if file.split('_')[0] != initial_seq_id:
                        seq_list = []
                        i = 0
                    else:
                        i += 1
                        seq_list.append(os.path.join(root, file))
                if i == seq_length:
                    file_list.append(seq_list)
                    seq_list = []
                    i = 0
        return file_list


class YOLODataset:
    def __init__(self, root, transform=None, target_transform=None, is_val=False,
                 label_file=None, is_eval=False, keep_random_transform=False, seq_limit=0):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_val = is_val
        self.is_eval = is_eval
        self.keep_random_transform = keep_random_transform
        self.seq_limit = seq_limit
        if is_val:
            self.subdir = 'val/img'
            self.image_sets_file = self.get_files(os.path.join(self.root, self.subdir), ext='.jpg')
            self.anno_sets_file = self.get_files(os.path.join(self.root, self.subdir), ext='.txt')
            self.shape_sets_file = self.get_files(os.path.join(self.root, self.subdir), ext='.shape')
        else:
            self.subdir = 'train/img'
            self.image_sets_file = self.get_files(os.path.join(self.root, self.subdir), ext='.jpg')
            self.anno_sets_file = self.get_files(os.path.join(self.root, self.subdir), ext='.txt')
            self.shape_sets_file = self.get_files(os.path.join(self.root, self.subdir), ext='.shape')

        self.ids = []
        for image_file in self.image_sets_file:
            self.ids.append(image_file.split('/')[-1].split('.')[0])

        # if the labels file exists, read in the class names
        label_file_name = self.root / "train/obj.names"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("YOLO Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND', 'dron')

        self._classes_names = self.class_names

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

        self.rnd = True
        self.new_seq_found = True
        self.prev_image_id = ''

    def __getitem__(self, index):
        if not self.is_eval:
            if self.keep_random_transform:
                if str(self.ids[index]).split('_')[0] != self.prev_image_id:
                    self.rng = preloaded_rng(1000)
                    self.prev_image_id = str(self.ids[index]).split('_')[0]
                    self.new_seq_found = True
                    print("DATASET: Generating new seed for random")

                local_rng = deepcopy(self.rng)
            else:
                if str(self.ids[index]).split('_')[0] != self.prev_image_id:
                    self.prev_image_id = str(self.ids[index]).split('_')[0]
                    self.new_seq_found = True
                    # print("DATASET: Next seq")

            boxes, labels = self._get_annotation(self.ids[index])
            image = self._read_image(index)
            original_size = image.shape
            if self.transform:
                if self.keep_random_transform and self.rnd:
                    image, boxes, labels = self.transform(image, boxes, labels, rng=local_rng)
                else:
                    image, boxes, labels = self.transform(image, boxes, labels)
            if self.target_transform:
                boxes, labels = self.target_transform(boxes, labels)
            return image, boxes, labels, original_size
        else:
            image_id = self.ids[index]
            boxes, labels = self._get_annotation(image_id)
            image = self._read_image(index)
            original_size = image.shape
            if self.transform:
                image, boxes, labels = self.transform(image, boxes, labels)
            if self.target_transform:
                boxes, labels = self.target_transform(boxes, labels)
            return image, boxes, labels, original_size

    def get_image(self, index, ret_id=False):
        image = self._read_image(index)
        image_id = self.ids[index]
        if self.transform:
            image, _ = self.transform(image)
        if ret_id:
            return image, image_id
        else:
            return image

    def get_annotation(self, index, ret_size=False):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id, ret_size=ret_size)

    def __len__(self):
        return len(self.image_sets_file)

    def _get_annotation(self, image_id, ret_size=False):
        annotation_file = os.path.join(os.path.join(self.root, self.subdir), image_id + '.txt')
        with open(os.path.join(os.path.join(self.root, self.subdir), image_id + '.shape')) as fp:
            line = fp.readline().split(' ')
            shape = (int(line[0]), int(line[1]), int(line[2]))

        if ret_size:
            img_size = (float(shape[1]), float(shape[0]))

        boxes = []
        labels = []
        with open(annotation_file) as fp:
            for cnt, line in enumerate(fp):
                line_ = line.split(' ')
                x1 = float(line_[1]) * float(shape[1])
                y1 = float(line_[2]) * float(shape[0])
                x2 = x1 + float(line_[3]) * float(shape[1])
                y2 = y1 + float(line_[4]) * float(shape[0])
                w = x2 - x1
                h = y2 - y1
                x1 -= w/2
                y1 -= h/2
                x2 -= w/2
                y2 -= h/2

                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[self._classes_names[int(line_[0]) + 1]])

        if ret_size:
            return (np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64), img_size)
        else:
            return (np.array(boxes, dtype=np.float32),
                    np.array(labels, dtype=np.int64))

    def _read_image(self, index):
        image_file = self.image_sets_file[index]
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_files(self, dir, ext='.jpg'):
        file_list = []
        if self.seq_limit > 0:
            cnt = 0
            seq = ''
        for root, dirs, files in os.walk(dir):
            for file in sorted(files):
                if file.endswith(ext):
                    if self.seq_limit > 0:
                        if cnt == 0:
                            seq = file.split('_')[0]

                    if self.seq_limit > 0 and cnt == self.seq_limit:
                        if file.split('_')[0] != seq:
                            seq = file.split('_')[0]
                            cnt = 0
                            file_list.append(os.path.join(root, file))
                    else:
                        file_list.append(os.path.join(root, file))

                    if self.seq_limit > 0 and not cnt == self.seq_limit:
                        cnt += 1
        return file_list

    def read_db(self):
        pass



