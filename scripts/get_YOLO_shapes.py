import os
import cv2


def get_files(dir, ext='.jpg'):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file))
    return file_list

root = '/path/to/mav_vid_dataset/'

sub_dir = 'val/img'
image_sets_file = get_files(os.path.join(root, sub_dir), ext='.jpg')

for file in image_sets_file:
    image_id = file.split('/')[-1].split('.')[0]
    print(image_id)
    try:
        with open(os.path.join(os.path.join(root, sub_dir), image_id + '.shape')) as fp:
            fp.readline()
    except:
        with open(os.path.join(os.path.join(root, sub_dir), image_id + '.shape'), 'a') as fp:
            image = cv2.imread(str(file))
            fp.write(str(image.shape[0]) + " " + str(image.shape[1]) + " " + str(image.shape[2]))
            print(str(image.shape[0]) + " " + str(image.shape[1]) + " " + str(image.shape[2]))

sub_dir = 'train/img'
image_sets_file = get_files(os.path.join(root, sub_dir), ext='.jpg')

for file in image_sets_file:
    image_id = file.split('/')[-1].split('.')[0]
    print(image_id)
    try:
        with open(os.path.join(os.path.join(root, sub_dir), image_id + '.shape')) as fp:
            fp.readline()
    except:
        with open(os.path.join(os.path.join(root, sub_dir), image_id + '.shape'), 'a') as fp:
            image = cv2.imread(str(file))
            fp.write(str(image.shape[0]) + " " + str(image.shape[1]) + " " + str(image.shape[2]))
            print(str(image.shape[0]) + " " + str(image.shape[1]) + " " + str(image.shape[2]))

