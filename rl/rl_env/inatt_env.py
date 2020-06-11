import gym
from gym import spaces
import pathlib

from models.inatt_models_ptl import InattModel
from utils.metrics import *
from utils.misc import sample_log_uniform
from dataloaders.vid_dataset import ImagenetDataset
from dataloaders.yolo_dataset import YOLODataset
from dataloaders.data_preprocessing import ValTransform, InverseValTransform, RLTrainTransform
from models.backbone import MatchPrior
from config import mobilenetv1_ssd_config
from utils.box_utils import nms
from utils.cropping_helper import *

import os

class InattEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, args, logging=None, n_actions=2, h_shape=(10, 10, 1024), history_shape=20, is_test=False,
                 val_dataset=True, random_train=False, cache_dir='', dynamic_gamma=False):
        super(InattEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(n_actions)
        # Example for using image as input:
        if dynamic_gamma:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(1, h_shape[0] * h_shape[1] * h_shape[2] + history_shape + 4 + 1), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(1, h_shape[0] * h_shape[1] * h_shape[2] + history_shape + 4), dtype=np.float32)

        self.is_test = is_test
        self.is_eval = False
        self.random_train = random_train
        self.dynamic_gamma = dynamic_gamma

        # Check if CUDA is available
        self.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.use_cuda else "cpu")
        if args.use_cuda and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if logging is not None:
                logging.info("Use Cuda.")
                
        # Actions history information
        self.nu = torch.zeros(history_shape).to(self.device)
        self.last_cropped_bbox = torch.FloatTensor([0,
                                                    0,
                                                    1,
                                                    1]).to(self.device)

        if not self.is_test:
            # Load model
            self.model = InattModel.load_from_checkpoint(args.trained_model, tags_csv=args.tags_csv)
            self.model.eval()
            self.model = self.model.to(self.device)

            # Load dataset
            eval_path = pathlib.Path(args.eval_dir)
            eval_path.mkdir(exist_ok=True)
            self.target_transform = MatchPrior(self.model.config.priors, self.model.config.center_variance, self.model.config.size_variance, 0.5)
            self.inverse_val_transform = InverseValTransform((1280, 720), self.model.config.image_mean, self.model.config.image_std)
            self.transform = RLTrainTransform(self.model.config.image_size, self.model.config.image_mean, self.model.config.image_std)

            # Datasets
            if args.dataset_type == "imagenet_vid":
                self.dataset = ImagenetDataset(args.dataset, args.dataset, transform=self.transform,
                                               target_transform=self.target_transform, is_val=val_dataset,
                                               keep_random_transform=self.random_train)

            elif args.dataset_type == "yolo":
                self.dataset = YOLODataset(args.dataset, transform=self.transform,
                                               target_transform=self.target_transform, is_val=val_dataset,
                                               keep_random_transform=self.random_train, seq_limit=400)

            print("RL_ENV: dataset images: " + str(len(self.dataset)))
            self.config = self.model.config
            self.num_classes = len(self.dataset._classes_names)
            self.true_case_stat, self.all_gb_boxes = group_annotation_by_class(self.dataset)

            # Others
            self.image_idx = 0
            if self.dynamic_gamma:
                self.nb_steps_gamma_upd = 10  # Number of sequences to sample gamma randomly
                self.seq_counter = 0
            self.boxes_old = torch.tensor([])
            self.faked_fmap_old = torch.tensor([])
            self.plot_image = False

        # Gamma definition
        self.gamma_r = args.lambda_0  # Parameter to control balance of mAP vs FPS

        # Metrics
        window_size = 100
        self.nb_cropped = torch.zeros(window_size).to(self.device)
        self.diff_loss = 0
        self.val_index = 0
        self.prob_threshold = 0.3
        self.results = []
        self.cache_dir = cache_dir


    def step(self, action):
        with torch.no_grad():
            # Compute history term
            self.nu = torch.cat((self.nu[1:], torch.FloatTensor([action]).to(self.device)))

            # Compute metrics
            self.nb_cropped = torch.cat((self.nb_cropped[1:], torch.FloatTensor([action]).to(self.device)))

            # Store previous LSTM state to be fair while computing losses
            h_prev = self.model.pred_dec.bottleneck_lstm1.hidden_state.clone()
            c_prev = self.model.pred_dec.bottleneck_lstm1.cell_state.clone()
            h_prev_prev = self.model.pred_dec.bottleneck_lstm1.prev_hidden_state.clone()
            c_prev_prev = self.model.pred_dec.bottleneck_lstm1.prev_cell_state.clone()

            # Make dataset in eval model
            if self.random_train:
                self.dataset.rnd = not self.is_eval
            else:
                self.dataset.rnd = False

            # Get image
            print("RL_ENV: Getting image no. " + str(self.image_idx))
            image, boxes, labels, original_size = self.dataset[self.image_idx]
            self.image_idx += 1
            if self.image_idx == len(self.dataset):
                print("RL_ENV: re-reading database")
                self.image_idx = 0
                self.dataset.read_db()  # TODO: Debug why the dataset modifies the labels after a second read

                if self.dynamic_gamma:
                    self.seq_counter = 0

            print("RL_ENV: Gamma value " + str(self.gamma_r))

            # Check if new sequence has been found
            if self.dynamic_gamma:
                if self.dataset.new_seq_found:
                    self.dataset.new_seq_found = False
                    self.seq_counter += 1
                    if self.seq_counter == self.nb_steps_gamma_upd:
                        self.seq_counter = 0
                        if not self.is_eval:
                            self.gamma_r = sample_log_uniform(low=0.01, high=2.0, size=1)[0]
                            print("RL_ENV: Sampling gamma from distribution")

            # Predict with model
            image = image.unsqueeze(0).to(self.device)
            boxes_batch = boxes.unsqueeze(0).to(self.device)
            labels_batch = labels.unsqueeze(0).to(self.device)

            if self.plot_image:
                img_draw, _, _ = self.inverse_val_transform(image[0], None, None)

            losses = []
            for a in range(self.action_space.n):
                # Start from correct state
                self.model.pred_dec.bottleneck_lstm1.hidden_state = h_prev.clone()
                self.model.pred_dec.bottleneck_lstm1.cell_state = c_prev.clone()
                self.model.pred_dec.bottleneck_lstm1.prev_hidden_state = h_prev_prev.clone()
                self.model.pred_dec.bottleneck_lstm1.prev_cell_state = c_prev_prev.clone()

                if a == 0:
                    scores, boxes = self.model.forward(image)
                    regression_loss, classification_loss = self.model.loss_criterion(self.model.pred_dec.confidences, self.model.pred_dec.locations, labels_batch, boxes_batch)
                    loss = regression_loss + classification_loss
                    losses.append(loss.item())
                    print("RL_ENV: L" + str(a) + ": " + str(loss.item()))

                    # Precompute loss difference
                    l0 = loss.item()

                    if action == a or self.boxes_old.nelement() == 0:
                        # Get input box
                        self.last_cropped_bbox = torch.FloatTensor([0,
                                                                    0,
                                                                    1,
                                                                    1]).to(self.device)

                        # Compute nms
                        boxes = boxes[0].cpu().detach()
                        scores = scores[0].cpu().detach()
                        boxes, labels, probs = self.compute_nms(scores, boxes, original_size)

                        # Compute mAP calculations
                        if self.is_eval:
                            # After prediction
                            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * self.val_index
                            self.val_index += 1

                            tmprslt = torch.cat([
                                indexes.clone().reshape(-1, 1).to(self.device),
                                labels.clone().reshape(-1, 1).float().to(self.device),
                                probs.clone().reshape(-1, 1).to(self.device),
                                (boxes.clone() + 1.0).to(self.device)  # matlab's indexes start from 1
                            ], dim=1)
                            if tmprslt.shape[0] > 0:
                                self.results.append(tmprslt)

                        if boxes.nelement() >= 4:
                            height, width, _ = original_size
                            boxes[:, 0] *= (self.model.config.image_size / width)
                            boxes[:, 1] *= (self.model.config.image_size / height)
                            boxes[:, 2] *= (self.model.config.image_size / width)
                            boxes[:, 3] *= (self.model.config.image_size / height)

                        # Update boxes old
                        boxes_temp = boxes[probs > self.prob_threshold]
                        fake_fmap_temp = self.model.feature_map

                        if self.plot_image:
                            cv2.rectangle(img_draw, (0, 0), (self.model.config.image_size, self.model.config.image_size),
                                          (255, 255, 255), 2)

                        if self.plot_image and boxes_temp.nelement() >= 4:
                            color = (36, 255, 12)

                            for j, box in enumerate(boxes):
                                if probs[j].item() > self.prob_threshold:  # Threshold
                                    x1 = int(box[0].cpu().item())
                                    y1 = int(box[1].cpu().item())
                                    x2 = int(box[2].cpu().item())
                                    y2 = int(box[3].cpu().item())

                                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(img_draw,
                                                self.dataset._classes_names[labels[j]] + " " + str(probs[j].cpu().item()),
                                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                color, 2)

                        # Store chosen "path"
                        self.h_temp = self.model.pred_dec.bottleneck_lstm1.hidden_state.clone()
                        self.c_temp = self.model.pred_dec.bottleneck_lstm1.cell_state.clone()
                        self.h_temp_prev = self.model.pred_dec.bottleneck_lstm1.prev_hidden_state.clone()
                        self.c_temp_prev = self.model.pred_dec.bottleneck_lstm1.prev_cell_state.clone()
                        print("RL_ENV: Followed action " + str(action) + " path")

                elif self.boxes_old.nelement():
                    # Get cropped bbox
                    adjustment_dict = self.build_cropped_bbox(self.boxes_old)
                    input_bbox = adjustment_dict['input_bbox']

                    # Get cropped image
                    image_cropped = image[:, :, int(input_bbox[1]):int(input_bbox[1] + input_bbox[3]),
                                    int(input_bbox[0]):int(input_bbox[0] + input_bbox[2])]

                    self.model.forward(image_cropped, full_processing=False)

                    # Fake feature map
                    faked_fmap = process_intermmediate_fmap(self.faked_fmap_old, 'prefaked_fmap_img_draw',
                                                            cropped_fmap=self.model.feature_map, adjustment_dict=adjustment_dict,
                                                            plot_image=False)

                    scores, boxes = self.model.forward(None, inter_tensor=faked_fmap)
                    regression_loss, classification_loss = self.model.loss_criterion(self.model.pred_dec.confidences, self.model.pred_dec.locations, labels_batch, boxes_batch)
                    loss = regression_loss + classification_loss
                    losses.append(loss.item())
                    print("RL_ENV: L" + str(a) + ": " + str(loss.item()))

                    # Precompute loss difference
                    l1 = loss.item()

                    if action == a:
                        # Get input box
                        self.last_cropped_bbox = torch.FloatTensor([input_bbox[0] / self.config.image_size,
                                                                    input_bbox[1] / self.config.image_size,
                                                                    input_bbox[2] / self.config.image_size,
                                                                    input_bbox[3] / self.config.image_size]).to(self.device)

                        # Compute nms
                        boxes = boxes[0].cpu().detach()
                        scores = scores[0].cpu().detach()
                        boxes, labels, probs = self.compute_nms(scores, boxes, original_size)

                        # Compute mAP calculations
                        if self.is_eval:
                            # After prediction
                            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * self.val_index
                            self.val_index += 1

                            tmprslt = torch.cat([
                                indexes.clone().reshape(-1, 1).to(self.device),
                                labels.clone().reshape(-1, 1).float().to(self.device),
                                probs.clone().reshape(-1, 1).to(self.device),
                                (boxes.clone() + 1.0).to(self.device)  # matlab's indexes start from 1
                            ], dim=1)
                            if tmprslt.shape[0] > 0:
                                self.results.append(tmprslt)

                        if boxes.nelement() >= 4:
                            height, width, _ = original_size
                            boxes[:, 0] *= (self.model.config.image_size / width)
                            boxes[:, 1] *= (self.model.config.image_size / height)
                            boxes[:, 2] *= (self.model.config.image_size / width)
                            boxes[:, 3] *= (self.model.config.image_size / height)

                        # Update boxes old
                        boxes_temp = boxes[probs > self.prob_threshold]
                        fake_fmap_temp = self.model.feature_map.clone()

                        if self.plot_image:
                            cv2.rectangle(img_draw, (int(input_bbox[0]), int(input_bbox[1])),
                                          (int(input_bbox[0] + input_bbox[2]), int(input_bbox[1] + input_bbox[3])),
                                          (255, 255, 255), 2)

                        if self.plot_image and boxes_temp.nelement() >= 4:
                            color = (36, 12, 255)

                            for j, box in enumerate(boxes):
                                if probs[j].item() > self.prob_threshold:  # Threshold
                                    x1 = int(box[0].cpu().item())
                                    y1 = int(box[1].cpu().item())
                                    x2 = int(box[2].cpu().item())
                                    y2 = int(box[3].cpu().item())

                                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(img_draw,
                                                self.dataset._classes_names[labels[j]] + " " + str(probs[j].cpu().item()),
                                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                color, 2)

                        # Store chosen "path"
                        self.h_temp = self.model.pred_dec.bottleneck_lstm1.hidden_state.clone()
                        self.c_temp = self.model.pred_dec.bottleneck_lstm1.cell_state.clone()
                        self.h_temp_prev = self.model.pred_dec.bottleneck_lstm1.prev_hidden_state.clone()
                        self.c_temp_prev = self.model.pred_dec.bottleneck_lstm1.prev_cell_state.clone()
                        print("RL_ENV: Followed action " + str(action) + " path")

            losses = np.asarray(losses)

            print("RL_ENV: Boxes nelements: " + str(boxes_temp.nelement()))

            if boxes_temp.nelement() >= 4:
                self.boxes_old = boxes_temp.clone()
                self.faked_fmap_old = fake_fmap_temp.clone()
                self.model.pred_dec.bottleneck_lstm1.hidden_state = self.h_temp.clone()
                self.model.pred_dec.bottleneck_lstm1.cell_state = self.c_temp.clone()
                self.model.pred_dec.bottleneck_lstm1.prev_hidden_state = self.h_temp_prev.clone()
                self.model.pred_dec.bottleneck_lstm1.prev_cell_state = self.c_temp_prev.clone()
            else:
                # Update the correct varibles due to action chosen
                print("RL_ENV: Boxes EMPTY: " + str(boxes_temp.nelement()))

            # Compute loss metrics
            if self.image_idx != 1:
                self.diff_loss = l0 - l1

            # Detach hidden
            self.model.detach_hidden()

            # Compute reward
            reward = self.compute_reward(losses, action)

            # Done
            done = False

            # Create observation
            observation = self.compute_observation()

            # Create info
            info = {"reward": reward}

            print("RL_ENV: Step called with action " + str(action) + " getting a reward of " + str(reward))

            if self.plot_image:
                cv2.imshow('img', img_draw)
                key = cv2.waitKey(0)
                if key == 27:  # if ESC is pressed, exit loop
                    cv2.destroyAllWindows()
                    self.plot_image = False

            return observation, reward, done, info

    def reset(self):
        self.boxes_old = torch.tensor([])
        self.faked_fmap_old = torch.tensor([])
        self.image_idx = 0
        self.val_index = 0
        self.results = []
        # self.plot_image = True  # Debug
        return self.compute_observation()  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close (self):
        pass

    def set_is_eval(self, is_eval):
        self.is_eval = is_eval

    def set_gamma_r(self, gamma):
        self.gamma_r = gamma

    def compute_map(self):
        if self.results:
            self.results = torch.cat(self.results)

            # OPTIONAL
            # Calculate and output mAP
            eval_path = pathlib.Path(self.cache_dir)
            for class_index, class_name in enumerate(self.dataset._classes_names):
                if class_index == 0: continue  # ignore background
                prediction_path = eval_path / f"det_test_{class_name}.txt"
                # Remove file before using it
                if os.path.exists(prediction_path):
                    os.remove(prediction_path)
                with open(prediction_path, "w") as f:
                    sub = self.results[self.results[:, 1] == class_index, :]
                    for i in range(sub.size(0)):
                        prob_box = sub[i, 2:].cpu().numpy()
                        image_id = self.dataset.ids[int(sub[i, 0])]
                        print(
                            image_id + " " + " ".join([str(v) for v in prob_box]),
                            file=f
                        )
            aps = []
            for class_index, class_name in enumerate(self.dataset._classes_names):
                if class_index == 0:
                    continue
                prediction_path = eval_path / f"det_test_{class_name}.txt"
                try:
                    ap = compute_average_precision_per_class(
                        self.true_case_stat[class_index],
                        self.all_gb_boxes[class_index],
                        prediction_path,
                        self.config.iou_threshold,
                        use_2007_metric=False
                    )
                    aps.append(ap)

                    # Remove after using it
                    if os.path.exists(prediction_path):
                        os.remove(prediction_path)
                except:
                    pass

            # Reseting variables
            self.val_index = 0
            self.results = []

            return sum(aps) / len(aps)
        else:
            print("RL_ENV: Results intermediate variable is empty")

            # Reseting variables
            self.val_index = 0
            self.results = []

            return -1

    def compute_observation(self, model=None, last_action=None, input_bbox=None):
        if model is None:
            with torch.no_grad():
                # Compute observation
                h = self.model.pred_dec.bottleneck_lstm1.hidden_state.clone().squeeze(0)
                c = self.model.pred_dec.bottleneck_lstm1.cell_state.clone().squeeze(0)
                h_prev = self.model.pred_dec.bottleneck_lstm1.prev_hidden_state.clone().squeeze(0)
                c_prev = self.model.pred_dec.bottleneck_lstm1.prev_cell_state.clone().squeeze(0)
                h_diff = h - h_prev
                c_diff = c - c_prev

                # Generate state observation
                obs = torch.cat((c, h, c_diff, h_diff), 0).transpose(0, 1).transpose(1, 2)

                if self.plot_image:
                    visualize_fmap(c.transpose(0, 1).transpose(1, 2), window_name='c')
                    visualize_fmap(h.transpose(0, 1).transpose(1, 2), window_name='h')
                    visualize_fmap(h_prev.transpose(0, 1).transpose(1, 2), window_name='h_prev')
                    visualize_fmap(c_prev.transpose(0, 1).transpose(1, 2), window_name='c_prev')
                    visualize_fmap(h_diff.transpose(0, 1).transpose(1, 2), window_name='h_diff')
                    visualize_fmap(c_diff.transpose(0, 1).transpose(1, 2), window_name='c_diff')

                obs = obs.reshape(-1)
                obs = torch.cat((obs, self.nu))
                obs = torch.cat((obs, self.last_cropped_bbox))
                if self.dynamic_gamma:
                    gamma = torch.FloatTensor([self.gamma_r]).to(self.device)
                    obs = torch.cat((obs, gamma))

                obs = obs.unsqueeze(0)

                obs = obs.clone().detach().cpu().numpy()
        else:
            with torch.no_grad():
                # Compute history term
                self.nu = torch.cat((self.nu[1:], torch.FloatTensor([last_action]).to(self.device)))

                self.last_cropped_bbox = torch.FloatTensor([input_bbox[0] / model.config.image_size,
                                                            input_bbox[1] / model.config.image_size,
                                                            input_bbox[2] / model.config.image_size,
                                                            input_bbox[3] / model.config.image_size]).to(self.device)

                # Compute observation
                h = model.pred_dec.bottleneck_lstm1.hidden_state.clone().squeeze(0)
                c = model.pred_dec.bottleneck_lstm1.cell_state.clone().squeeze(0)
                h_prev = model.pred_dec.bottleneck_lstm1.prev_hidden_state.clone().squeeze(0)
                c_prev = model.pred_dec.bottleneck_lstm1.prev_cell_state.clone().squeeze(0)
                h_diff = h - h_prev
                c_diff = c - c_prev

                # Generate state observation
                obs = torch.cat((c, h, c_diff, h_diff), 0).transpose(0, 1).transpose(1, 2) # TODO: Is it necessary to clone before the detach?

                # visualize_fmap(c.transpose(0, 1).transpose(1, 2), window_name='c')
                # visualize_fmap(h.transpose(0, 1).transpose(1, 2), window_name='h')
                # visualize_fmap(h_prev.transpose(0, 1).transpose(1, 2), window_name='h_prev')
                # visualize_fmap(c_prev.transpose(0, 1).transpose(1, 2), window_name='c_prev')
                # visualize_fmap(h_diff.transpose(0, 1).transpose(1, 2), window_name='h_diff')
                # visualize_fmap(c_diff.transpose(0, 1).transpose(1, 2), window_name='c_diff')

                obs = obs.reshape(-1)
                obs = torch.cat((obs, self.nu))
                obs = torch.cat((obs, self.last_cropped_bbox))

                if self.dynamic_gamma:
                    print("INFO: Gamma has been set to " + str(self.gamma_r))
                    gamma = torch.FloatTensor([self.gamma_r]).to(self.device)
                    obs = torch.cat((obs, gamma))

                obs = obs.unsqueeze(0)

                obs = obs.clone().detach().cpu().numpy()
        return obs

    def compute_reward(self, losses, action):
        # Compute minimum loss
        min_loss = losses.min()
        min_idex = losses.argmin()
        print("RL_ENV: Lmin (L" + str(min_idex) + "): " + str(min_loss))

        try:
            # Compute loss difference
            reward = min_loss - losses[action]

            # Include speed gamma incentive
            if action != 0:
                reward += self.gamma_r
        except:
            print("RL_ENV: First reward should be")
            reward = 0

        return reward


    def compute_nms(self, scores, boxes, original_size):
        height, width, _ = original_size
        # height = height.item()
        # width = width.item()
        # original_size = (width, height)

        # Filtering by confidence threshold?
        prob_threshold = 0.01

        # Compute prediction with NMS
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = nms(box_probs, "hard",
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.config.iou_threshold,
                                      sigma=0.5,
                                      top_k=-1,
                                      candidate_size=200)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        else:
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, 0] *= width
            picked_box_probs[:, 1] *= height
            picked_box_probs[:, 2] *= width
            picked_box_probs[:, 3] *= height
            return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

    def build_cropped_bbox(self, boxes, cropping_overlap=0, bbox_increase_factor=0.05):
        x1 = boxes[:, 0].min(dim=0)[0].item()
        y1 = boxes[:, 1].min(dim=0)[0].item()
        x2 = boxes[:, 2].max(dim=0)[0].item()
        y2 = boxes[:, 3].max(dim=0)[0].item()

        # Last box is stored
        cx = x1 + (x2 - x1) / 2
        cy = y1 + (y2 - y1) / 2
        w = x2 - x1
        h = y2 - y1
        s = max(w, h) * (1.0 + bbox_increase_factor)
        focus_box = np.array([cx - s / 2, cy - s / 2,
                              cx + s / 2, cy + s / 2])

        # Saturate focus box
        focus_box[0] = saturate_img_coordinate(focus_box[0], self.model.config.image_size)
        focus_box[1] = saturate_img_coordinate(focus_box[1], self.model.config.image_size)
        focus_box[2] = saturate_img_coordinate(focus_box[2], self.model.config.image_size)
        focus_box[3] = saturate_img_coordinate(focus_box[3], self.model.config.image_size)

        # Extract proper bbox based on feature map size
        bbox = (focus_box[0], focus_box[1], focus_box[2] - focus_box[0], focus_box[3] - focus_box[1])  # (x,y,w,h)

        # In case the bbox has w or h equal to zero
        if bbox[2] == 0 or bbox[3] == 0:
            bbox = (0, 0, self.model.config.image_size, self.model.config.image_size)  # (x,y,w,h)

        adjustment_dict = adjust_bbox(bbox, cropping_overlap, self.model.config.image_size, self.model.config.image_size, self.model.pred_enc)

        return adjustment_dict # x, y, w, h


