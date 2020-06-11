## Adaptive Inattentional Framework for Video Object Detection with Reward-Conditional Training #
This is the original implementation of the article <url_link>

![alt text](main_image.png)

Citation bibtex here.

### Abstract ###
Recent object detection studies have been focused on video sequences, mostly due to the increasing demand of industrial applications. Although single-image architectures achieve remarkable results in terms of accuracy, they do not take advantage of particular properties of the video sequences and usually require high parallel computational resources, such as desktop GPUs. In this work, an inattentional framework is proposed, where the object context in video frames is dynamically reused in order to reduce the computation overhead. The context features corresponding to keyframes are fused into a synthetic feature map, which is further refined using temporal aggregation with ConvLSTMs. Furthermore, an inattentional policy has been learned to adaptively balance the accuracy and the amount of context reused. The inattentional policy has been learned under the reinforcement learning paradigm, and using our novel reward-conditional training scheme, which allows for policy training over a whole distribution of reward functions and enables the selection of a unique reward function at inference time. Our framework shows outstanding results on platforms with reduced parallelization capabilities, such as CPUs, achieving an average latency reduction up to 2.09x, and obtaining FPS rates similar to their equivalent GPU platform, at the cost of a 0.09 mAP reduction.

### System Requirements ###

- Python 3.6+
- PyTorch 1.4.0
- Torchvision
- OpenCV
- Tensorflow 1.14.0
- Stable Baselines 2.10.0
- PyTorch Lightning 0.7.3

### Installation Instructions (with Anaconda) ###

- Install [Anaconda](https://docs.anaconda.com/anaconda/install/).
- Create and [activate an environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
- Install the system dependencies (follow the instructions for [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/guide/install.html))
- Clone this repo by `git clone https://github.com/alejodosr/adaptive-inattention`

(a specific version of a package can be installed with `pip install package=="0.0.0"`)

### Datasets ###

This framework has been validated with Imagent VID 2015 and a custom Multirotor Aerial Vehicles (MAV-VID) dataset:
- Imagenet VID 2015 dataset can be downloaded [here](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php). Once downloaded, use these scripts, in this order, `scripts/get_list.py` and `scripts/get_seq_list.py` (modify the path inside the script).
- MAV-VID dataset can be downloaded [here](https://bitbucket.org/alejodosr/mav-vid-dataset). The MAV-VID dataset has been annotated with [YOLO annotations style](https://github.com/AlexeyAB/Yolo_mark).

If you want to use your custom dataset, YOLO-style annotations can be used and `scripts/get_YOLO_shapes.py` has to be used to generate extra annotations.

### Additional notes ###
Every script has several options and parameters. These are some important ones:
- For selecting MobileNetV2 backbone use `--backbone mobilenetv2` (if not included, by default MobileNetV1 is selected).
- For MAV-VID dataset (or Yolo style dataset) use `--dataset_type yolo` (if not included, by default Imagenet VID 2015 dataset is selected).
- For CPU evaluation use `--use_cuda False` (by default GPU is used if present)
- Extended data augmentation is included for training (see article).  To disable it use `--crop_prob 0.0` or, conversely, you can increase the probability `0.x` of including a synthetic feature map in the batch with `--crop_prob 0.x`

## Training ##
### Train the Inattentional Model ###

Train the Fature Extractor:

`CUDA_VISIBLE_DEVICES=0 python train_inattentional_model.py --net backbone --datasets /path/to/dataset/ --batch_size 16 --num_epochs 200 --width_mult 1 --cache_path /path/to/cache/folder --validation_epochs 3 --lr 0.0001 --scheduler plateau`

or with MobileNetV2 backbone:

`CUDA_VISIBLE_DEVICES=0 python train_inattentional_model.py --net backbone --datasets /path/to/dataset/ --batch_size 16 --num_epochs 200 --width_mult 1 --cache_path /path/to/cache/folder --validation_epochs 3 --lr 0.0001 --scheduler plateau --backbone mobilenetv2`

Train the Temporal Aggregator (ConvLSTM):

```CUDA_VISIBLE_DEVICES=0 python train_inattentional_model.py --net lstm --datasets /path/to/dataset/ --batch_size 50 --num_epochs 200 --width_mult 1 --cache_path /home/alejo/Downloads/cache --pretrained /path/to/checkpoint/checkpoint.ckpt  --freeze_net --crop_prob 0.05```

or with MobileNetV2 backbone:

`CUDA_VISIBLE_DEVICES=0 python train_inattentional_model.py --net lstm --datasets /path/to/dataset/ --batch_size 50 --num_epochs 200 --width_mult 1 --cache_path /home/alejo/Downloads/cache --pretrained /path/to/checkpoint/checkpoint.ckpt --freeze_net --crop_prob 0.05 --backbone mobilenetv2`


### Train the Inattentional Policy (Reinforcement Learning) ###

`CUDA_VISIBLE_DEVICES=0 python train_rl_inattentional_policy.py --net lstm --dataset /path/to/dataset/ --trained_model /path/to/checkpoint/checkpoint.ckpt --label_file /path/to/labels/vid-model-labels.txt --tags_csv /path/to/tags/meta_tags.csv`

## Evaluation ##
### Evaluate Trained Model (with Random Policy) ###

`CUDA_VISIBLE_DEVICES=0 python evaluate_inattentional_policy.py --net lstm --policy baseline --dataset /path/to/dataset/ --trained_model /path/to/checkpoint/checkpoint.ckpt --label_file /path/to/labels/vid-model-labels.txt --tags_csv /path/to/tags/meta_tags.csv --iter 5 --prob 0.5 --use_cuda True`

### Evaluate Trained Model (with Inattentional Policy) ###

``CUDA_VISIBLE_DEVICES=0 python evaluate_inattentional_policy.py --net lstm --policy rl_ppo2 --dataset /path/to/dataset/ --trained_model /path/to/checkpoint/checkpoint.ckpt --label_file /path/to/labels/vid-model-labels.txt --label_file /path/to/tags/meta_tags.csv --rl_path /path/to/rl/checkpoint --iter 5 --lambda_0 1.2 --use_cuda True``

## Acknowledgement ##
