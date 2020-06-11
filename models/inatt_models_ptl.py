""" Models in pytorch lighting """

from torch.utils.data import DataLoader
from models.multibox_loss import MultiboxLoss
from config import mobilenetv1_ssd_config
import models.backbone
import models.backbone_conv_lstm
import models.backbone_v2
import models.backbone_conv_lstm_v2
from dataloaders.data_preprocessing import TrainTransform, ValTransform, InverseValTransform, PredictionTransform
from utils.metrics import *
from numpy import random
from utils.cropping_helper import *
from utils.box_utils import nms
from utils import box_utils

import pytorch_lightning as ptl
import os
import pathlib

#debug
import cv2


class InattModel(ptl.LightningModule):
    """
    Arguments:
        hparams : arguments for the construction of the model
            hparams.gpu_id : id of the GPU to be used
            hparams.use_cuda : True is CUDA is to be used
            hparams.num_classes : Number fo classesof the dataset
            hparams.width_mult : Mobilenet width multiplier
            hparams.batch_size : Bach size
            hparams.num_workers : Number of workersfor data loading
        logging: a logger
    """

    def __init__(self, hparams, logging=None):
        super(InattModel, self).__init__()

        self.hparams = hparams
        self.logging = logging
        self.device = torch.device("cuda:" + str(hparams.gpu_id) if torch.cuda.is_available() and hparams.use_cuda else "cpu")
        if hparams.net == 'backbone':
            self.config = mobilenetv1_ssd_config
            self.priors = self.config.priors.to(self.device)
            if hparams.backbone == 'mobilenetv1':
                self.pred_enc = models.backbone.MobileNetV1(num_classes=hparams.num_classes, alpha=hparams.width_mult)
                self.pred_dec = models.backbone.SSD(num_classes=hparams.num_classes, alpha=hparams.width_mult, config=self.config, device=self.device)
                if self.logging is not None:
                    self.logging.info("Backbone: mobilenetv1")
                else:
                    print("INFO: backbone mobilenetV1")
            elif hparams.backbone == 'mobilenetv2':
                self.pred_enc = models.backbone_v2.MobileNetV2(num_classes=hparams.num_classes, alpha=hparams.width_mult)
                self.pred_dec = models.backbone_v2.SSD(num_classes=hparams.num_classes, alpha=hparams.width_mult, config=self.config, device=self.device)
                if self.logging is not None:
                    self.logging.info("Backbone: mobilenetv2")
                else:
                    print("INFO: backbone mobilenetV2")
            else:
                if self.logging is not None:
                    self.logging.fatal("Backbone not implemented")

            # Optimizer selection
            if self.logging is not None:
                self.logging.info(f"Adam with learning rate: {self.hparams.lr}")
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

            # Adaptive lr
            if self.hparams.scheduler == 'plateau':
                patience = 10
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, verbose=True)
                if self.logging is not None:
                    self.logging.info(f"ReduceLROnPlateau with patience: {patience}")
            elif self.hparams.scheduler == 'exponential':
                if self.logging is not None:
                    self.logging.info(f"ExponentialLR with gamma: {self.hparams.gamma_sched}")
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.hparams.gamma_sched,
                                                                           last_epoch=-1)

        elif hparams.net == 'lstm':
            self.config = mobilenetv1_ssd_config
            self.priors = self.config.priors.to(self.device)
            if hparams.backbone == 'mobilenetv1':
                self.pred_enc = models.backbone_conv_lstm.MobileNetV1(num_classes=hparams.num_classes, alpha=hparams.width_mult)
                self.pred_dec = models.backbone_conv_lstm.SSD(num_classes=hparams.num_classes, alpha=hparams.width_mult, config=self.config, batch_size=hparams.batch_size, device=self.device)
                if self.logging is not None:
                    self.logging.info("Backbone: mobilenetv1")
                else:
                    print("INFO: backbone mobilenetV1")
            elif hparams.backbone == 'mobilenetv2':
                self.pred_enc = models.backbone_conv_lstm_v2.MobileNetV2(num_classes=hparams.num_classes, alpha=hparams.width_mult)
                self.pred_dec = models.backbone_conv_lstm_v2.SSD(num_classes=hparams.num_classes, alpha=hparams.width_mult, config=self.config, batch_size=hparams.batch_size, device=self.device)
                if self.logging is not None:
                    self.logging.info("Backbone: mobilenetv2")
                else:
                    print("INFO: backbone mobilenetV2")
            else:
                if self.logging is not None:
                    self.logging.fatal("Backbone not implemented")

            # Optimizer selection
            if self.logging is not None:
                # self.logging.info(f"RMSprop with learning rate: {self.hparams.lr}")
                self.logging.info(f"Adam with learning rate: {self.hparams.lr}")

            # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.003,
            #                                         weight_decay=hparams.weight_decay, momentum=hparams.momentum)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=hparams.weight_decay)

            # Adaptive lr
            patience = 10
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, verbose=True)
            if self.logging is not None:
                self.logging.info(f"ReduceLROnPlateau with patience: {patience}")

        self.batch = hparams.batch_size
        self.num_workers = hparams.num_workers

        self.val_transform = ValTransform(self.config.image_size, self.config.image_mean, self.config.image_std)
        self.prediction_transform = PredictionTransform(self.config.image_size, self.config.image_mean, self.config.image_std)
        self.inverse_val_transform = InverseValTransform((1280, 720), self.config.image_mean, self.config.image_std)

        # Loss function
        self.loss_criterion = MultiboxLoss(self.config.priors, iou_threshold=0.5, neg_pos_ratio=10,
                                          center_variance=0.1, size_variance=0.2, device=self.device)

        # For loss and mAP calculation
        self.accum_loss = 0
        self.train_index = 0
        self.val_index = 0
        self.accum_val_loss = 0
        self.results = []

        # For debugging
        self.plot_image = False

    def forward(self, seq, full_processing=True, inter_tensor=None):
        """
        Arguments:
            seq : a tensor used as input to the model
        Returns:
            confidences and locations of predictions made by model
        """

        if inter_tensor is not None:
            confidences, locations = self.pred_dec(inter_tensor)
            self.feature_map = inter_tensor
            return confidences, locations
        else:
            if full_processing:
                self.feature_map = self.pred_enc(seq)
                confidences, locations = self.pred_dec(self.feature_map)
                return confidences, locations
            else:
                self.feature_map = self.pred_enc(seq)

                return torch.tensor([]), torch.tensor([])

    def training_step(self, batch, batch_nb):
        if self.hparams.net == 'backbone':
            # Compute loss
            images, boxes, labels, _ = batch  # gt

            # Debug
            # img_debug, _, _ = self.inverse_val_transform(images[0], None, None)
            # img_debug = cv2.resize(img_debug, (160, 160))
            # cv2.imshow("img_debug", img_debug)
            # key = cv2.waitKey(1)

            confidence, locations = self.forward(images)

            regression_loss, classification_loss = self.loss_criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

            # Metrics computation
            self.accum_loss += loss.item()
            self.train_index += 1

        elif self.hparams.net == 'lstm':
            # Detach hidden states from graph
            self.detach_hidden()
            
            # Compute loss
            images, boxes, labels = batch  # gt

            loss = 0
            crop_flag=False
            for image, box, label in zip(images, boxes, labels):
                image = image.to(self.device)
                box = box.to(self.device)
                label = label.to(self.device)
                
                if(random.rand() < self.hparams.crop_prob and crop_flag): #use cropped image as training sample, also we skip first image crops to make sure that self.featuremap makes sense
                    faked_fmap = self.feature_map.clone() #create a copy of previous fmap in order to merge wit cropped fmap, we detach it from the graph because we dont want the gradients to go back this branch
                    n = image.shape[0]
                    for i in range(n):
                        #calculate crop
                        current_image=torch.unsqueeze( image[i,:,:,:],0)
                        boxes_old = torch.squeeze(last_box[i,:,:])
                        labels_old = torch.squeeze(last_label[i,:])
                        boxes_old = box_utils.convert_locations_to_boxes(torch.unsqueeze(boxes_old,0), self.pred_dec.priors, self.pred_dec.config.center_variance, self.pred_dec.config.size_variance)
                        boxes_old = box_utils.center_form_to_corner_form(boxes_old)
                        boxes_old = torch.squeeze(boxes_old)

                        boxes_old= boxes_old[labels_old>0,:]
                        labels_old = labels_old[labels_old>0]


                        boxes_old*= self.config.image_size #scale boxes to image size
                        if(boxes_old.shape[0] >0): 
                            #now we have to calculate a bbox that envolves all bboxes

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
                            s = max(w, h) * (1.0 + self.hparams.bbox_increase_factor)
                            focus_box = np.array([cx - s / 2, cy - s / 2,cx + s / 2, cy + s / 2])
                            # Saturate focus box
                            focus_box[0] = saturate_img_coordinate(focus_box[0], self.config.image_size)
                            focus_box[1] = saturate_img_coordinate(focus_box[1], self.config.image_size)
                            focus_box[2] = saturate_img_coordinate(focus_box[2], self.config.image_size)
                            focus_box[3] = saturate_img_coordinate(focus_box[3], self.config.image_size)

                            # Debug
                            # img_debug, _, _ = self.inverse_val_transform(image[i], None, None)
                            # cv2.rectangle(img_debug, (int(focus_box[0]), int(focus_box[1])),
                            #                           (int(focus_box[2]), int(focus_box[3])), (255, 255, 0), 2)
                            # cv2.imshow("img_debug", img_debug)
                            # key = cv2.waitKey(0)

                            bbox = (focus_box[0], focus_box[1], focus_box[2] - focus_box[0], focus_box[3] - focus_box[1])  # (x,y,w,h)
                            adjustment_dict = adjust_bbox(bbox, 0, self.config.image_size, self.config.image_size,self.pred_enc)
                            # Crop image
                            input_bbox = adjustment_dict['input_bbox']
                            if input_bbox[3] <= 16 or input_bbox[2] <= 16:
                                image_cropped = current_image
                                self.forward(image_cropped, full_processing=False)  # we can access the fmap as model.feature_map, but its going to be cropped... beteter to acces later, when its padded
                                faked_fmap[i, :, :, :] = self.feature_map
                            else:
                                image_cropped = current_image[:,:,int(input_bbox[1]):int(input_bbox[1] + input_bbox[3]),int(input_bbox[0]):int(input_bbox[0] + input_bbox[2])]
                                #print(image_cropped.shape)
                                self.forward(image_cropped, full_processing=False) # we can access the fmap as model.feature_map, but its going to be cropped... beteter to acces later, when its padded
                                current_faked_fmap = process_intermmediate_fmap(torch.unsqueeze(faked_fmap[i,:,:,:],0), 'prefaked_fmap_img_draw',cropped_fmap=self.feature_map, adjustment_dict=adjustment_dict,plot_image=False)
                                faked_fmap[i,:,:,:]= torch.unsqueeze(current_faked_fmap,0)
                        else:#if boxes are empty, we have to processs the whole image
                            self.forward(image_cropped, full_processing=False) # we can access the fmap as model.feature_map, but its going to be cropped... beteter to acces later, when its padded
                            faked_fmap[i,:,:,:]=self.feature_map
                    confidence, locations = self.forward(image, full_processing=False, inter_tensor=faked_fmap)


                else:#train as usual
                    # Debug
                    # img_debug, _, _ = self.inverse_val_transform(image[0], None, None)
                    # cv2.imshow("img_debug", img_debug)
                    # key = cv2.waitKey(0)
                    
                    crop_flag=True
                    confidence, locations = self.forward(image)
                regression_loss, classification_loss = self.loss_criterion(confidence, locations, label, box)
                loss += regression_loss + classification_loss
                last_box=box
                last_label=label

            # Metrics computation
            self.accum_loss += loss.item()
            self.train_index += 1

        tensorboard_logs = {'train_loss': loss, 'epoch':  self.current_epoch}

        return {'loss': loss, 'log': tensorboard_logs}

    def on_epoch_end(self):
        avg_epoch_loss = self.accum_loss / self.train_index
        self.accum_loss = 0
        self.train_index = 0
        if self.hparams.scheduler == 'plateau':
            self.lr_scheduler.step(avg_epoch_loss)
        elif self.hparams.scheduler == 'exponential':
            self.lr_scheduler.step()
            print("INFO: Last learning rate scheduled: " + str(self.lr_scheduler.get_last_lr()))

    def validation_step(self, batch, batch_nb):
        if self.hparams.net == 'lstm':
            # Detach hidden states from graph
            self.detach_hidden()

            if not int(self.pred_dec.bottleneck_lstm1.hidden_state.shape[0]) == 1 or not int(self.pred_dec.bottleneck_lstm1.cell_state.shape[0]) == 1:
                # Adjust hidden state due to batch size
                (h, c) = self.pred_dec.bottleneck_lstm1.cell.init_hidden(1, hidden=self.pred_dec.bottleneck_lstm1.hidden_channels, shape=(10, 10))
                self.pred_dec.bottleneck_lstm1.hidden_state = h
                self.pred_dec.bottleneck_lstm1.cell_state = c

        # OPTIONAL
        images, boxes_batch, labels_batch, original_size = batch  # gt

        scores, boxes = self.forward(images)
        regression_loss, classification_loss = self.loss_criterion(self.pred_dec.confidences, self.pred_dec.locations, labels_batch, boxes_batch)
        loss = regression_loss + classification_loss
        self.accum_val_loss += loss.item()

        # Apply inverse transform
        boxes = boxes[0]
        scores = scores[0]
        image, _, _ = self.inverse_val_transform(images[0], None, None)

        # height, width, _ = image.shape
        height, width, _ = original_size
        height = height.item()
        width = width.item()
        original_size = (width, height)

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
            boxes, labels, probs = torch.tensor([]), torch.tensor([]), torch.tensor([])
        else:
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, 0] *= width
            picked_box_probs[:, 1] *= height
            picked_box_probs[:, 2] *= width
            picked_box_probs[:, 3] *= height
            boxes, labels, probs = picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

        if self.plot_image:
            img_draw = image.copy()
            img_draw = cv2.resize(img_draw, original_size)

            for j, box in enumerate(boxes):
                if probs[j].item() > 0.01:   # Threshold
                    x1 = int(box[0].cpu().item())
                    y1 = int(box[1].cpu().item())
                    x2 = int(box[2].cpu().item())
                    y2 = int(box[3].cpu().item())
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (36, 255, 12), 2)
                    cv2.putText(img_draw, self.val_dataset._classes_names[labels[j]] + " " + str(probs[j].cpu().item()), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
            cv2.imshow("img", img_draw)
            key = cv2.waitKey(0)
            if key == 27:  # if ESC is pressed, exit loop
                self.plot_image = False
                cv2.destroyAllWindows()

        # After prediction
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * self.val_index
        self.val_index += 1

        tmprslt = torch.cat([
            indexes.reshape(-1, 1).to(self.device),
            labels.reshape(-1, 1).float().to(self.device),
            probs.reshape(-1, 1).to(self.device),
            (boxes + 1.0).to(self.device)  # matlab's indexes start from 1
        ], dim=1)
        if tmprslt.shape[0] > 0:
            self.results.append(tmprslt)

        tensorboard_logs = {'val_loss': loss}

        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        # For debugging
        self.plot_image = False

        if self.hparams.net == 'lstm':
            # Adjust hidden state size
            (h, c) = self.pred_dec.bottleneck_lstm1.cell.init_hidden(self.hparams.batch_size,
                                                                     hidden=self.pred_dec.bottleneck_lstm1.hidden_channels,
                                                                     shape=(10, 10))
            self.pred_dec.bottleneck_lstm1.hidden_state = h
            self.pred_dec.bottleneck_lstm1.cell_state = c

        if self.results:
            self.results = torch.cat(self.results)

            # OPTIONAL
            # Calculate and output mAP
            eval_path = pathlib.Path(self.cache_dir)
            for class_index, class_name in enumerate(self.val_dataset._classes_names):
                if class_index == 0: continue  # ignore background
                prediction_path = eval_path / f"det_test_{class_name}.txt"
                # Remove file before using it
                if os.path.exists(prediction_path):
                    os.remove(prediction_path)
                with open(prediction_path, "w") as f:
                    sub = self.results[self.results[:, 1] == class_index, :]
                    for i in range(sub.size(0)):
                        prob_box = sub[i, 2:].cpu().numpy()
                        image_id = self.val_dataset.ids[int(sub[i, 0])]
                        print(
                            image_id + " " + " ".join([str(v) for v in prob_box]),
                            file=f
                        )
            aps = []
            print("\n\nAverage Precision Per-class:")
            for class_index, class_name in enumerate(self.val_dataset._classes_names):
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
                    print(f"{class_name}: {ap}")

                    # Remove after using it
                    if os.path.exists(prediction_path):
                        os.remove(prediction_path)
                except:
                    print(f"{class_name}: NaN")

            print(f"\nAverage Precision Across All Classes:{sum(aps) / len(aps)}")

            val_loss_avg = self.accum_val_loss / self.val_index
            tensorboard_logs = {'mAP': sum(aps) / len(aps), 'val_loss_avg': val_loss_avg}

            # Reseting variables
            self.val_index = 0
            self.accum_val_loss = 0
            self.results = []

            return {'val_loss': val_loss_avg, 'map': sum(aps) / len(aps), 'log': tensorboard_logs}
        else:
            self.logging.info(f"Results intermediate variable is empty")

            # Reseting variables
            self.val_index = 0
            self.accum_val_loss = 0
            self.results = []

            return {}

    def set_val_dataset(self, val_dataset):
        self.val_dataset = val_dataset
        self.true_case_stat, self.all_gb_boxes = group_annotation_by_class(self.val_dataset)

    def set_train_dataset(self, train_dataset):
        self.train_dataset = train_dataset

    def set_cache_dir(self, cache_dir):
        self.cache_dir = cache_dir

    def set_encoder_decoder(self, pred_enc, pred_dec):
        """
        Arguments:
            pred_enc : an object of MobilenetV1 class
            pred_dec : an object of SSD class
        """
        self.pred_encoder = pred_enc
        self.pred_decoder = pred_dec

    def adjust_device(self, device):
        self.device = device
        self.priors = self.config.priors.to(device)
        self.pred_dec.adjust_device(device)

    def predict(self, image, full_processing=True, inter_tensor=None):

        height, width, _ = image.shape

        if inter_tensor is None:
            image = self.prediction_transform(image, resize=full_processing)
            images = image.unsqueeze(0)
            images = images.to(self.device)

        with torch.no_grad():
            if inter_tensor is not None:
                scores, boxes = self.forward(None, full_processing=full_processing, inter_tensor=inter_tensor)
            else:
                scores, boxes = self.forward(images, full_processing=full_processing)

        if not full_processing:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), 0

        # Compute metrics
        boxes = boxes[0]
        scores = scores[0]

        # Move to cpu?
        boxes = boxes.to(torch.device("cpu"))
        scores = scores.to(torch.device("cpu"))

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

            return torch.tensor([]), torch.tensor([]), torch.tensor([]), 0
        else:
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, 0] *= width
            picked_box_probs[:, 1] *= height
            picked_box_probs[:, 2] *= width
            picked_box_probs[:, 3] *= height

            return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4], 0

    def configure_optimizers(self):
        return [self.optimizer]

    def detach_hidden(self):
        """
        Detaches hidden state and cell state of all the LSTM layers from the graph
        """
        self.pred_dec.bottleneck_lstm1.hidden_state.detach_()
        self.pred_dec.bottleneck_lstm1.cell_state.detach_()

    @ptl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          self.batch, num_workers=self.num_workers, shuffle=True)
    @ptl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          1, num_workers=1, shuffle=False)

    # @ptl.data_loader
    # def test_dataloader(self):
    #     return DataLoader(self.val_dataset,
    #                       1, num_workers=1, shuffle=False)
    def compute_nms(self, scores, boxes, original_size):
        height, width, _ = original_size
        boxes=boxes.cpu().detach()
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
    def compute_scores(self,labels):
        labels = labels.cpu().detach()
        scores=torch.zeros(  (labels.shape[0],self.hparams.num_classes))
        for i in range(1,labels.shape[0]):
            scores[i,labels[i]]=1
        return scores
