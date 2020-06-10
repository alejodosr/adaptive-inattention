## Adaptive Inattentional Framework for Video Object Detection with Reward-Conditional Training #
This is the original implementation of the article <url_link>

![alt text](main_image.png)

Citation bibtex here.

### System Requirements ###

- Python 3.6+
- PyTorch 1.4.0
- Torchvision
- OpenCV
- Tensorflow 1.14.0
- Stable Baselines 2.10.0
- PyTorch Lightning 0.7.3

### Installation Instructions (with Anaconda) ###

Install conda.

### Train the Inattentional Model ###

Train the Fature Extractor:

`CUDA_VISIBLE_DEVICES=0 python train_inattentional_model.py --net backbone --datasets /path/to/dataset/ --batch_size 16 --num_epochs 200 --width_mult 1 --cache_path /path/to/cache/folder --validation_epochs 3 --lr 0.0001 --scheduler plateau`

or with MobileNetV2 backbone:

`CUDA_VISIBLE_DEVICES=0 python train_inattentional_model.py --net backbone --datasets /path/to/dataset/ --batch_size 16 --num_epochs 200 --width_mult 1 --cache_path /path/to/cache/folder --validation_epochs 3 --lr 0.0001 --scheduler plateau --backbone mobilenetv2`

Train the Temporal Aggregator (convLSTM):

`CUDA_VISIBLE_DEVICES=0 python train_inattentional_model.py --net lstm --datasets /path/to/dataset/ --batch_size 50 --num_epochs 200 --width_mult 1 --cache_path /home/alejo/Downloads/cache --pretrained /home/alejo/py_workspace/high-res-mobile-object-detection/checkpoints/lstm1/lightning_logs/last_uav_vid_v2_version_26/checkpoints/epoch\=23.ckpt --backbone mobilenetv2 --dataset_type yolo --freeze_net --crop_prob 0.05`

or with MobileNetV2 backbone:

`CUDA_VISIBLE_DEVICES=0 python train_inattentional_model.py --net lstm --datasets /path/to/dataset/ --batch_size 50 --num_epochs 200 --width_mult 1 --cache_path /home/alejo/Downloads/cache --pretrained /home/alejo/py_workspace/high-res-mobile-object-detection/checkpoints/lstm1/lightning_logs/last_uav_vid_v2_version_26/checkpoints/epoch\=23.ckpt --backbone mobilenetv2 --dataset_type yolo --freeze_net --crop_prob 0.05 --backbone mobilenetv2`


### Train the Inattentional Policy (Reinforcement Learning) ###

`CUDA_VISIBLE_DEVICES=0 python train_rl_inattentional_policy.py --net lstm --dataset /path/to/dataset/ --trained_model /path/to/checkpoint/checkpoint.ckpt --label_file /path/to/labels/vid-model-labels.txt --tags_csv /path/to/tags/meta_tags.csv`
### Evaluate Trained Model (with Random Policy) ###

`CUDA_VISIBLE_DEVICES=0 python evaluate_inattentional_policy.py --net lstm --policy baseline --dataset /path/to/dataset/ --trained_model /path/to/checkpoint/checkpoint.ckpt --label_file /path/to/labels/vid-model-labels.txt --tags_csv /path/to/tags/meta_tags.csv --iter 3 --prob 0.5 --use_cuda True`

### Evaluate Trained Model (with Inattentional Policy) ###

``CUDA_VISIBLE_DEVICES=0 python evaluate_inattentional_policy.py --net lstm --policy rl_ppo2 --dataset /path/to/dataset/ --trained_model /path/to/checkpoint/checkpoint.ckpt --label_file /path/to/labels/vid-model-labels.txt --label_file /path/to/tags/meta_tags.csv --rl_path /path/to/rl/checkpoint --iter 3 --lambda_0 1.2 --use_cuda True``