import numpy as np
import torch
from utils import box_utils


def compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()


def compute_voc2007_average_precision(precision, recall):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap


def compute_average_precision_per_class(num_true_cases, gt_boxes,
                                        prediction_file, iou_threshold, use_2007_metric):
    """ Computes average precision per class
    """
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if (image_id, max_arg) not in matched:
                    true_positive[i] = 1
                    matched.add((image_id, max_arg))
                else:
                    false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return compute_voc2007_average_precision(precision, recall)
    else:
        return compute_average_precision(precision, recall)


def group_annotation_by_class(dataset, ran=None):
    """ Groups annotations of dataset by class
    """
    true_case_stat = {}
    all_gt_boxes = {}
    if ran is not None:
        r0 = ran[0]
        r1 = ran[1]
    else:
        r0 = 0
        r1 = len(dataset)

    for ii in range(r0, r1):
        image_id, annotation = dataset.get_annotation(ii)
        try:
            gt_boxes, classes = annotation
        except:
            gt_boxes, classes, is_difficult = annotation

        gt_boxes = torch.from_numpy(gt_boxes)
        for jj in range(0, len(classes)):
            class_index = int(classes[jj])
            gt_box = gt_boxes[jj]
            true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1
            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes

def saturate_img_coordinate(x, max):
    if x < 0:
        y = 0
    elif x > max:
        y = max
    else:
        y = x
    return y

class ClassificationMetrics:
    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tp_tn = 0.0
        self.total = 0.0

        self.prec = 0.0
        self.rec = 0.0

    def accuracy(self):
        return self.tp_tn / self.total

    def precision(self):
        self.prec = self.tp / (self.tp + self.fp + 1E-9)
        return self.prec

    def recall(self):
        self.rec = self.tp / (self.tp + self.fn + 1E-9)
        return self.rec

    def f1(self):
        return (2 * self.prec * self.rec) / (self.prec + self.rec + 1E-9)

    def store_metrics(self, out, labels):
        outputs = np.argmax(out, axis=1)
        tp_tn = outputs == labels
        fp_fn = outputs != labels
        self.fp += np.sum(outputs[fp_fn] == 1.0)
        self.tp += np.sum(outputs[tp_tn] == 1.0)
        self.fn += np.sum(outputs[fp_fn] == 0.0)
        self.tp_tn += np.sum(outputs == labels)
        self.total += float(labels.size)

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tp_tn = 0.0
        self.total = 0.0
