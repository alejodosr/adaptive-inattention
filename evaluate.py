#!/usr/bin/python3
"""Script for evaluation of trained model on the dataset.
Few global variables defined here are explained:
Global Variables
----------------
args : dict
	Has all the options for changing various variables of the model as well as parameters for evaluation
dataset : ImagenetDataset (torch.utils.data.Dataset, For more info see datasets/vid_dataset.py)

"""
from models.inatt_models_ptl import HRMobileVOD
from dataloaders.vid_dataset import ImagenetDataset
from dataloaders.yolo_dataset import YOLODataset
from config import mobilenetv1_ssd_config
from utils.metrics import *
from utils.misc import str2bool
import argparse
import pathlib
import logging
import cv2
import os

parser = argparse.ArgumentParser(description="MVOD Evaluation on VID dataset")
parser.add_argument('--net', default="lstm9",
                    help="The network architecture, it should be of backbone, lstm.")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--tags_csv", type=str)
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--dataset_type", default="imagenet_vid", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument('--gpu_id', default=0, type=int,
                    help='The GPU id to be used')
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="temp", type=str, help="The directory to store evaluation results.")
parser.add_argument('--width_mult', default=1.0, type=float,
                    help='Width Multiplifier for network')
parser.add_argument("--minival", type=str2bool, default=False)
parser.add_argument('--frames_per_seq', default=10, type=int,
                    help='')

# Debug
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_classes', default=31, type=int,
                    help='')
parser.add_argument('--lr', '--learning-rate', default=0, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay')

args = parser.parse_args()

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

if __name__ == '__main__':
    # Active for plotting images
    plot_image = True
    initial_img_idx = 0
    nb_img_evaluated = 1e9

    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "imagenet_vid":
        dataset = ImagenetDataset(args.dataset, args.dataset, is_val=True, is_eval=True)
    elif args.dataset_type == "yolo":
        dataset = YOLODataset(args.dataset, is_val=True, is_eval=True)

    end_img_idx = min(initial_img_idx + nb_img_evaluated, len(dataset))
    config = mobilenetv1_ssd_config
    num_classes = len(dataset._classes_names)
    print("INFO: number of images is " + str(len(dataset)))
    true_case_stat, all_gb_boxes = group_annotation_by_class(dataset, ran=(initial_img_idx, end_img_idx))

    # Check if CUDA is available
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.use_cuda else "cpu")
    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")

    # Load model
    model = HRMobileVOD.load_from_checkpoint(args.trained_model, tags_csv=args.tags_csv)

    model.eval()
    model = model.to(device)

    results = []
    if args.minival:
        frame_count = 0
        prev_image_id = ''

    for i in range(initial_img_idx, end_img_idx):
        print("process image", i)
        image, image_id = dataset.get_image(i, ret_id=True)
        if args.minival:
            if args.dataset_type == "imaget_vid":
                print("INFO: " + image_id.split('/')[0])
                if image_id.split('/')[0] != prev_image_id:
                    prev_image_id = image_id.split('/')[0]
                    frame_count = 0
            elif args.dataset_type == "yolo":
                print("INFO: " + image_id.split('_')[0])
                if image_id.split('_')[0] != prev_image_id:
                    prev_image_id = image_id.split('_')[0]
                    frame_count = 0

            if frame_count > args.frames_per_seq:
                for b in range(num_classes):
                    try:
                        num_gts_bboxes = all_gb_boxes[b][image_id].shape[0]
                        all_gb_boxes[b] = removekey(all_gb_boxes[b], image_id)
                        true_case_stat[b] -= num_gts_bboxes
                        print("INFO: Removing element " + image_id)
                    except:
                        continue
                continue

        boxes, labels, probs, t = model.predict(image)

        if args.minival:
            frame_count += 1
        if plot_image:
            img_draw = image.copy()
            for j, box in enumerate(boxes):
                if probs[j].item() > 0.3:   # Threshold
                    x1 = int(box[0].cpu().item())
                    y1 = int(box[1].cpu().item())
                    x2 = int(box[2].cpu().item())
                    y2 = int(box[3].cpu().item())
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (36, 255, 12), 2)
                    cv2.putText(img_draw, dataset._classes_names[labels[j]] + " " + str(probs[j].cpu().item()), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
            img_draw = cv2.resize(img_draw, (320, 320))
            cv2.imshow("img_draw", img_draw)
            key = cv2.waitKey(1)
            if key == 27:  # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                plot_image = False
        # Detaching is not necessary while evaluating
        # if args.net != 'backbone':
        #     model.detach_hidden()
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        tmprslt = torch.cat([
            indexes.reshape(-1, 1).to(device),
            labels.reshape(-1, 1).float().to(device),
            probs.reshape(-1, 1).to(device),
            (boxes + 1.0).to(device)  # matlab's indexes start from 1
        ], dim=1)
        if (tmprslt.shape[0] > 0):
            results.append(tmprslt)

    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        # Remove file before using it
        if os.path.exists(prediction_path):
            os.remove(prediction_path)
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].cpu().numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        try:
            ap = compute_average_precision_per_class(
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                prediction_path,
                args.iou_threshold,
                use_2007_metric=False
            )
            aps.append(ap)
            print(f"{class_name}: {ap}")

            # Remove after using it
            if os.path.exists(prediction_path):
                os.remove(prediction_path)
        except:
            print(f"{class_name}: NaN")

    print(f"\nAverage Precision Across All Classes:{sum(aps) / len(aps)}")
