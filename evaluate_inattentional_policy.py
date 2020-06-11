#!/usr/bin/python3
"""Script for evaluation of trained model on the dataset.
Few global variables defined here are explained:
Global Variables
----------------
args : dict
	Has all the options for changing various variables of the model as well as parameters for evaluation
dataset : ImagenetDataset (torch.utils.data.Dataset, For more info see datasets/vid_dataset.py)

"""
from models.inatt_models_ptl import InattModel
from dataloaders.data_preprocessing import TrainTransform, ValTransform, RandTransform
from dataloaders.vid_dataset import ImagenetDataset
from dataloaders.yolo_dataset import YOLODataset
from config import mobilenetv1_ssd_config
from utils.misc import str2bool
from utils.cropping_helper import *
import argparse
import pathlib
import logging
from utils.metrics import *
import cv2
import os
from stable_baselines import PPO2
from rl.rl_env.inatt_env import InattEnv
from rl.rl_agent.custom_policies import BaselinePolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from dataloaders.transforms import preloaded_rng
from copy import deepcopy

parser = argparse.ArgumentParser(description="MVOD Evaluation on VID dataset")
parser.add_argument('--net', default="lstm5",
                    help="The network architecture, it should be of backbone, lstm.")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--tags_csv", type=str)
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--dataset_type", default="imagenet_vid", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument('--policy', default=None, type=str,
                    help='type of policy')
parser.add_argument('--rl_path', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument('--gpu_id', default=0, type=int,
                    help='The GPU id to be used')
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="temp", type=str, help="The directory to store evaluation results.")
parser.add_argument('--width_mult', default=1.0, type=float,
                    help='Width Multiplifier for network')
parser.add_argument("--normalize_env", type=str2bool, default=True)
parser.add_argument("--lambda_0", type=float, default=0.5, help="Lambda_0 modulates mAP in RL agent")
parser.add_argument("--prob", type=float, default=0.5, help="Modulates the % cropped in random baseline")
parser.add_argument('--iter', default=1, type=int, help='Number of iterations over the dataset')
parser.add_argument('--load_state_dict', default='', type=str, help='Load state dict model')
args = parser.parse_args()

def initialize_model(net):
    """ Loads learned weights from pretrained checkpoint model
    Arguments:
        net : object of MobileVOD
    """
    if args.load_state_dict:
        logging.info(f"Loading weights from pretrained mobilenetv1 netwok ({args.load_state_dict})")
        pretrained_net_dict = torch.load(args.load_state_dict)
        model_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_net_dict = {k: v for k, v in pretrained_net_dict.items() if
                               k in model_dict and model_dict[k].shape == pretrained_net_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_net_dict)
        # 3. load the new state dict
        net.load_state_dict(model_dict)


if __name__ == '__main__':

    # Parameters
    use_prev_fmap = True  # Decide to use previous feature map or to use zeros instead
    val_dataset = True   # Test with validation or training dataset
    rand_transform = False  # Decide to use random transforms to images for evaluation
    dynamic_lambda_0 = True   # Is the model trained on a distribution of rewards?
    map_sub_classes = True  # Provide the map of subclasses
    sub_classes = ['bicycle', 'airplane', 'lion', 'car', 'fox', 'bird', 'monkey']
    initial_img_idx = 0
    nb_img_evaluated = 100e3
    cropping_overlap = 0.0
    bbox_increase_factor = 0.1
    max_bbox_size = 1.0  # Wrt the whole image
    prob_threshold = 0.3  # Threshold for min prediction
    prob_baseline = args.prob

    if args.policy == 'baseline':
        policy = BaselinePolicy(prob_baseline)

    elif args.policy == 'rl_ppo2':
        # Create and wrap the environment
        h_shape = (10, 10, 1024)  # Shape of the hidden state of the lstm network
        history_shape = 20  # Number of past actions to be tracked
        env = InattEnv(args, h_shape=h_shape, history_shape=history_shape, is_test=True, dynamic_gamma=dynamic_lambda_0)
        if dynamic_lambda_0:
            print("INFO: Gamma has been set to " + str(args.lambda_0))
            env.lambda_0 = args.lambda_0

        env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

        # Create RL policy
        policy = PPO2.load(os.path.join(args.rl_path, "best_agent.zip"), env=env, verbose=1)

        if args.normalize_env:
            if os.path.exists(os.path.join(args.rl_path, 'vecnormalize.pkl')):
                env = VecNormalize.load(os.path.join(args.rl_path, 'vecnormalize.pkl'), env)
                env.training = False
                env.norm_reward = False
            else:
                raise Exception("Normalization parameters not found")

        print("INFO: Loaded model " + os.path.join(args.rl_path, "best_agent.zip"))
    else:
        raise Exception("Policy type not recognized")

    # Active for plotting images
    plot_image = True

    if plot_image:
        def on_trackbar(val):
            if args.policy == 'baseline':
                policy.prob = val / 100
            elif args.policy == 'rl_ppo2':
                env.set_attr("gamma_r", val / 100, 0)
                # print(env.get_attr("gamma_r", 0)[0])

        # Create trackbar
        title_window = 'img'
        cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Lambda 0', title_window, 0, 300, on_trackbar)

    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    if args.dataset_type == "imagenet_vid":
        dataset = ImagenetDataset(args.dataset, args.dataset, is_val=val_dataset, is_eval=True)
    elif args.dataset_type == "yolo":
        dataset = YOLODataset(args.dataset, is_val=val_dataset, is_eval=True, seq_limit=300)

    config = mobilenetv1_ssd_config
    print("INFO: Number of images " + str(len(dataset)))
    num_classes = len(dataset._classes_names)
    # Compute limits
    end_img_idx = min(initial_img_idx + nb_img_evaluated, len(dataset))
    true_case_stat, all_gb_boxes = group_annotation_by_class(dataset, ran=(initial_img_idx, end_img_idx))
    
    transform = RandTransform(config.image_size, config.image_mean, config.image_std)

    # Check if CUDA is available
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.use_cuda else "cpu")
    print("INFO: Using device: " + str(device))
    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")

    # Load model
    model = InattModel.load_from_checkpoint(args.trained_model, tags_csv=args.tags_csv)

    # Initialize with pretrained
    initialize_model(model)

    model.eval()
    model = model.to(device)
    model.adjust_device(device)

    mean_map = []
    if map_sub_classes:
        sub_mean_map = []

    frame_time = 0
    for it in range(args.iter):
        results = []
        iou_accum = 0.0
        boxes_old = torch.tensor([])
        boxes = torch.tensor([0, 0])
        last_action = 0
        actions_executed = []
        input_bbox = [0, 0, config.image_size, config.image_size]

        if rand_transform:
            prev_image_id = ''

        for i in range(initial_img_idx, end_img_idx):
            print("\nINFO: Process image", i)

            if rand_transform:
                image, image_id = dataset.get_image(i, ret_id=True)
            else:
                image = dataset.get_image(i)

            # Store sizes
            previous_img_size = image.shape # (h, w)
            current_size = (model.config.image_size, model.config.image_size)

            if rand_transform:
                gt_class_img = -1
                gt_boxes_img = torch.zeros([])
                for b in range(num_classes):
                    try:
                        gt_boxes_img = all_gb_boxes[b][image_id].clone().numpy()
                        gt_class_img = b
                    except:
                        continue

                if args.dataset_type == 'imagenet_vid':
                    if image_id.split('/')[0] != prev_image_id:
                        rng = preloaded_rng(1000)
                        prev_image_id = image_id.split('/')[0]
                elif args.dataset_type == 'yolo':
                    if image_id.split('_')[0] != prev_image_id:
                        rng = preloaded_rng(1000)
                        prev_image_id = image_id.split('_')[0]

                local_rng = deepcopy(rng)
                image, trans_boxes, _ = transform(image, gt_boxes_img, gt_boxes_img, rng=local_rng)
                image = cv2.UMat(image).get()  # Found on github due to errors

                # Update sizes
                rand_img_size = image.shape  # (h, w)

                for s in range(trans_boxes.shape[0]):
                    trans_boxes[s][0] = saturate_img_coordinate(trans_boxes[s][0], previous_img_size[1])
                    trans_boxes[s][1] = saturate_img_coordinate(trans_boxes[s][1], previous_img_size[0])
                    trans_boxes[s][2] = saturate_img_coordinate(trans_boxes[s][2], previous_img_size[1])
                    trans_boxes[s][3] = saturate_img_coordinate(trans_boxes[s][3], previous_img_size[0])

                rand_boxes = torch.from_numpy(trans_boxes).float()
                if rand_boxes.shape[0]:
                    rand_boxes[:, 0] *= (previous_img_size[1] / rand_img_size[1])
                    rand_boxes[:, 1] *= (previous_img_size[0] / rand_img_size[0])
                    rand_boxes[:, 2] *= (previous_img_size[1] / rand_img_size[1])
                    rand_boxes[:, 3] *= (previous_img_size[0] / rand_img_size[0])

                all_gb_boxes[gt_class_img][image_id] = rand_boxes.clone()

                # image = cv2.resize(image, (previous_img_size[1], previous_img_size[0]))

                # for s in range(trans_boxes.shape[0]):
                #     print(int(rand_boxes[s][0]))
                #     print(int(trans_boxes[s][1]))
                #     print(int(trans_boxes[s][2]))
                #     print(int(trans_boxes[s][3]))
                #
                # x1 = int(rand_boxes[0][0].item())
                # y1 = int(rand_boxes[0][1].item())
                # x2 = int(rand_boxes[0][2].item())
                # y2 = int(rand_boxes[0][3].item())
                #
                # cv2.rectangle(image, (x1, y1), (x2, y2), (36, 13, 12), 2)
                # cv2.imshow('img_rand', image)
                # key = cv2.waitKey(0)
                # if key == 27:  # if ESC is pressed, exit loop
                #     cv2.destroyAllWindows()
                #     plot_image = False

            if plot_image:
                img_draw = image.copy()
            image = cv2.resize(image, current_size)

            # Retrieve action
            if args.policy != 'baseline':
                arguments = (model, last_action, input_bbox)
                obs = env.env_method('compute_observation', *arguments)
                if args.normalize_env:
                    obs = env.normalize_obs(obs)
                last_action = policy.predict(obs)[0][0]
            else:
                last_action = policy.predict()

            # Append actions
            actions_executed.append(last_action)

            # Action executed
            print("INFO: Action executed " + str(last_action))

            # If we are detecting less boxes than before
            if last_action == 0 or boxes_old.nelement() < 4:
                print("INFO: Full image prediction")

                # Detect full image
                boxes, labels, probs, _ = model.predict(image)

                # Last input bbox
                input_bbox = [0, 0, config.image_size, config.image_size]

                if boxes[probs > prob_threshold].nelement() >= 4:
                    boxes_old = boxes[probs > prob_threshold]
                    faked_fmap = model.feature_map

                if plot_image:
                    window_name = 'img'
                    color = (36, 255, 12)
                    # img_draw = image.copy()

                    for j, box in enumerate(boxes):
                        if probs[j].item() > prob_threshold:  # Threshold
                            x1 = int(box[0].cpu().item() * (previous_img_size[1] / current_size[1]))
                            y1 = int(box[1].cpu().item() * (previous_img_size[0] / current_size[0]))
                            x2 = int(box[2].cpu().item() * (previous_img_size[1] / current_size[1]))
                            y2 = int(box[3].cpu().item() * (previous_img_size[0] / current_size[0]))

                            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img_draw, dataset._classes_names[labels[j]] + " " + str(probs[j].cpu().item()),
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        color, 2)
                    if plot_image:
                        if args.policy == 'rl_ppo2':
                            cv2.putText(img_draw, 'Lambda_0: ' + str(env.get_attr("gamma_r", 0)[0]),
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                        (255, 255, 255), 2)
                        elif args.policy == 'baseline':
                            cv2.putText(img_draw, 'Prob: ' + str(args.prob),
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                        (255, 255, 255), 2)
                        cv2.imshow(window_name, img_draw)
                        key = cv2.waitKey(frame_time)
                        if key == 27:  # if ESC is pressed, exit loop
                            cv2.destroyAllWindows()
                            plot_image = False
                        if key == 115:  # if ESC is pressed, exit loop
                            frame_time = 0
                        else:
                            frame_time = 1

            else:
                print("INFO: Cropped image prediction")

                # update boxes old
                # boxes_old = boxes

                # Show intermmediate feature map
                # process_intermmediate_fmap(faked_fmap, plot_image=plot_image)

                # if plot_image:
                #     img_draw = image.copy()

                # Build global bounding box
                x1 = boxes_old[:, 0].min(dim=0)[0].item()
                y1 = boxes_old[:, 1].min(dim=0)[0].item()
                x2 = boxes_old[:, 2].max(dim=0)[0].item()
                y2 = boxes_old[:, 3].max(dim=0)[0].item()

                # Last box is stored
                square_side = max(x2 - x1, y2 - y1)
                cx = x1 + (x2 - x1) / 2
                cy = y1 + (y2 - y1) / 2
                w = x2 - x1
                h = y2 - y1
                s = max(w, h) * (1.0 + bbox_increase_factor)
                focus_box = np.array([cx - s / 2, cy - s / 2,
                                      cx + s / 2, cy + s / 2])

                # Saturate focus box
                focus_box[0] = saturate_img_coordinate(focus_box[0], model.config.image_size)
                focus_box[1] = saturate_img_coordinate(focus_box[1], model.config.image_size)
                focus_box[2] = saturate_img_coordinate(focus_box[2], model.config.image_size)
                focus_box[3] = saturate_img_coordinate(focus_box[3], model.config.image_size)

                # Extract proper bbox based on feature map size
                bbox = (focus_box[0], focus_box[1], focus_box[2] - focus_box[0], focus_box[3] - focus_box[1])  # (x,y,w,h)

                adjustment_dict = adjust_bbox(bbox, cropping_overlap, model.config.image_size, model.config.image_size,
                                              model.pred_enc)

                # Crop image
                input_bbox = adjustment_dict['input_bbox']
                image_cropped = image[int(input_bbox[1]):int(input_bbox[1] + input_bbox[3]),
                                int(input_bbox[0]):int(input_bbox[0] + input_bbox[2])].copy()

                if max(input_bbox[3], input_bbox[2]) <= (max_bbox_size * config.image_size):
                    print("INFO: Bounding box IS within the permitted size: " + str(max(input_bbox[3], input_bbox[2])) + " (max. " + str(max_bbox_size * config.image_size) + ")")
                    bbox_within_bounds = True
                else:
                    print("INFO: Bounding box is NOT within the permitted size: " + str(max(input_bbox[3], input_bbox[2])) + " (max. " + str(max_bbox_size * config.image_size) + ")")
                    bbox_within_bounds = False

                # Get cropped feature map
                _, _, _, p = model.predict(image_cropped, full_processing=False)

                if plot_image:
                    cv2.imshow("img_cropped", image_cropped)

                # Show inermmediate feature map
                # process_intermmediate_fmap(model.feature_map, 'cropped_fmap_img_draw', plot_image=plot_image)
                if not use_prev_fmap:
                    faked_fmap = torch.zeros(faked_fmap.shape).to(device)
                faked_fmap = process_intermmediate_fmap(faked_fmap, 'prefaked_fmap_img_draw',
                                                        cropped_fmap=model.feature_map, adjustment_dict=adjustment_dict,
                                                        plot_image=False)
                # process_intermmediate_fmap(faked_fmap, 'faked_fmap_img_draw')

                # Predict with new feature map
                boxes, labels, probs, p = model.predict(image, inter_tensor=faked_fmap)

                if boxes[probs > prob_threshold].nelement() >= 4:
                    boxes_old = boxes[probs > prob_threshold]
                    faked_fmap = model.feature_map

                if plot_image:
                    # img_draw = image.copy()
                    img_background = np.zeros(img_draw.shape, dtype=image.dtype)
                    input_bbox_0 = input_bbox[0] * (previous_img_size[1] / current_size[1])
                    input_bbox_1 = input_bbox[1] * (previous_img_size[0] / current_size[0])
                    input_bbox_2 = input_bbox[2] * (previous_img_size[1] / current_size[1])
                    input_bbox_3 = input_bbox[3] * (previous_img_size[0] / current_size[0])

                    img_background[int(input_bbox_1):int(input_bbox_1 + input_bbox_3),
                                    int(input_bbox_0):int(input_bbox_0 + input_bbox_2)] = img_draw[int(input_bbox_1):int(input_bbox_1 + input_bbox_3),
                                int(input_bbox_0):int(input_bbox_0 + input_bbox_2)]
                    window_name = 'img'
                    color = (36, 12, 255)
                    for j, box in enumerate(boxes):
                        if probs[j].item() > prob_threshold:  # Threshold
                            x1 = int(box[0].cpu().item() * (previous_img_size[1] / current_size[1]))
                            y1 = int(box[1].cpu().item() * (previous_img_size[0] / current_size[0]))
                            x2 = int(box[2].cpu().item() * (previous_img_size[1] / current_size[1]))
                            y2 = int(box[3].cpu().item() * (previous_img_size[0] / current_size[0]))

                            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img_draw, dataset._classes_names[labels[j]] + " " + str(probs[j].cpu().item()),
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        color, 2)

                    img_final = cv2.addWeighted(img_background, 0.7, img_draw, 0.3, 0)
                    img_final[int(input_bbox_1):int(input_bbox_1 + input_bbox_3),
                    int(input_bbox_0):int(input_bbox_0 + input_bbox_2)] = img_draw[int(input_bbox_1):int(input_bbox_1 + input_bbox_3),
                    int(input_bbox_0):int(input_bbox_0 + input_bbox_2)]
                    if args.policy == 'rl_ppo2':
                        cv2.putText(img_draw, 'Lambda_0: ' + str(env.get_attr("gamma_r", 0)[0]),
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                    (255, 255, 255), 2)
                    elif args.policy == 'baseline':
                        cv2.putText(img_draw, 'Prob: ' + str(args.prob),
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                    (255, 255, 255), 2)
                    cv2.imshow(window_name, img_final)
                    key = cv2.waitKey(frame_time)
                    if key == 27:  # if ESC is pressed, exit loop
                        cv2.destroyAllWindows()
                        plot_image = False
                    if key == 115:  # if ESC is pressed, exit loop
                        frame_time = 0
                    else:
                        frame_time = 1

            # Boxes
            if boxes.shape[0]:
                boxes[:, 0] *= (previous_img_size[1] / current_size[1])
                boxes[:, 1] *= (previous_img_size[0] / current_size[0])
                boxes[:, 2] *= (previous_img_size[1] / current_size[1])
                boxes[:, 3] *= (previous_img_size[0] / current_size[0])

            if rand_transform:
                if plot_image:
                    img_draw = cv2.resize(image, (previous_img_size[1], previous_img_size[0]))

                    window_name = 'final results'

                    for j, box in enumerate(boxes):
                        if probs[j].item() > prob_threshold:  # Threshold
                            x1 = int(box[0].cpu().item())
                            y1 = int(box[1].cpu().item())
                            x2 = int(box[2].cpu().item())
                            y2 = int(box[3].cpu().item())

                            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img_draw, dataset._classes_names[labels[j]] + " " + str(probs[j].cpu().item()),
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        color, 2)

                    color = (36, 12, 14)
                    for j, box in enumerate(rand_boxes):
                        x1 = int(box[0].cpu().item())
                        y1 = int(box[1].cpu().item())
                        x2 = int(box[2].cpu().item())
                        y2 = int(box[3].cpu().item())

                        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

                    cv2.imshow(window_name, img_draw)


            if args.net != 'backbone':
                model.detach_hidden()

            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
            tmprslt = torch.cat([
                indexes.reshape(-1, 1).to(device),
                labels.reshape(-1, 1).float().to(device),
                probs.reshape(-1, 1).to(device),
                (boxes + 1.0).to(device)  # matlab's indexes start from 1
            ], dim=1)
            if (tmprslt.shape[0] > 0):
                results.append(tmprslt)

            # Boxes
            if boxes.shape[0]:
                boxes[:, 0] /= (previous_img_size[1] / current_size[1])
                boxes[:, 1] /= (previous_img_size[0] / current_size[0])
                boxes[:, 2] /= (previous_img_size[1] / current_size[1])
                boxes[:, 3] /= (previous_img_size[0] / current_size[0])

        # # Calculate te avg IOU
        # iou_avg = iou_accum / len(dataset)
        # print("Avg IoU wrt to cropped bounding boxes: " + str(iou_avg))

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
        if map_sub_classes:
            sub_aps = []
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
                if map_sub_classes:
                    for sc in sub_classes:
                        if class_name == sc:
                            sub_aps.append(ap)

                # Remove after using it
                if os.path.exists(prediction_path):
                    os.remove(prediction_path)
            except:
                print(f"{class_name}: NaN")

            # Calculate map and average it
            mean_map.append(sum(aps) / len(aps))

            if map_sub_classes:
                sub_mean_map.append(sum(sub_aps) / len(sub_aps))

    print("Policy: " + str(args.policy))
    print("Number of iterations: " + str(args.iter))
    print("Image size: " + str(config.image_size) + "x" + str(config.image_size))
    print("Percentage of cropped actions executed: " + str(np.asarray(actions_executed).sum() * 100.0 / len(actions_executed)))
    if dynamic_lambda_0 and args.policy != 'baseline':
        print("Gamma: " + str(args.lambda_0))
    print(f"Average Precision Across All Classes:{np.asarray(mean_map).sum() / len(mean_map)}")
    if map_sub_classes:
        print("Average Precision Across Sub Classes(" + str(len(sub_aps)) + "): " + str(np.asarray(sub_mean_map).sum() / len(sub_mean_map)))