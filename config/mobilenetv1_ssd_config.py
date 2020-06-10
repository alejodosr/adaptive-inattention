#!/usr/bin/python3
"""Script containing configuration parameters for SSD
"""

import numpy as np

from utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


# image_size = 720  # RL trained first with this
# image_mean = np.array([127, 127, 127])  # RGB layout
# image_std = 128.0
# iou_threshold = 0.5
# center_variance = 0.1
# size_variance = 0.2
#
# specs = [
#     SSDSpec(45, 16, SSDBoxSizes(135, 236), [2, 3]),
#     SSDSpec(23, 31, SSDBoxSizes(236, 337), [2, 3]),
#     SSDSpec(12, 60, SSDBoxSizes(337, 438), [2, 3]),
#     SSDSpec(6, 120, SSDBoxSizes(438, 540), [2, 3]),
#     SSDSpec(3, 240, SSDBoxSizes(540, 641), [2, 3]),
#     SSDSpec(2, 360, SSDBoxSizes(641, 742), [2, 3])
# ]
#
# priors = generate_ssd_priors(specs, image_size)

# image_size = 1080 # Priors meant for 320
# image_mean = np.array([127, 127, 127])  # RGB layout
# image_std = 128.0
# iou_threshold = 0.45
# center_variance = 0.1
# size_variance = 0.2
#
# specs = [
#     SSDSpec(68, 16, SSDBoxSizes(60, 105), [2, 3]),
#     SSDSpec(34, 32, SSDBoxSizes(105, 150), [2, 3]),
#     SSDSpec(17, 64, SSDBoxSizes(150, 195), [2, 3]),
#     SSDSpec(9, 100, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(5, 150, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(3, 300, SSDBoxSizes(285, 330), [2, 3])
# ]
#
# priors = generate_ssd_priors(specs, image_size)

# image_size = 720 # Priors meant for 320
# image_mean = np.array([127, 127, 127])  # RGB layout
# image_std = 128.0
# iou_threshold = 0.45
# center_variance = 0.1
# size_variance = 0.2
#
# specs = [
#     SSDSpec(45, 16, SSDBoxSizes(60, 105), [2, 3]),
#     SSDSpec(23, 32, SSDBoxSizes(105, 150), [2, 3]),
#     SSDSpec(12, 64, SSDBoxSizes(150, 195), [2, 3]),
#     SSDSpec(6, 100, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(3, 150, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(2, 300, SSDBoxSizes(285, 330), [2, 3])
# ]
#
# priors = generate_ssd_priors(specs, image_size)

image_size = 320
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(20, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)

# image_size = 640
# image_mean = np.array([127, 127, 127])  # RGB layout
# image_std = 128.0
# iou_threshold = 0.45
# center_variance = 0.1
# size_variance = 0.2
#
# specs = [
#     SSDSpec(40, 16, SSDBoxSizes(60, 105), [2, 3]),
#     SSDSpec(20, 32, SSDBoxSizes(105, 150), [2, 3]),
#     SSDSpec(10, 64, SSDBoxSizes(150, 195), [2, 3]),
#     SSDSpec(5, 100, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(3, 150, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(2, 300, SSDBoxSizes(285, 330), [2, 3])
# ]
#
# priors = generate_ssd_priors(specs, image_size)


# image_size = 160
# image_mean = np.array([127, 127, 127])  # RGB layout
# image_std = 128.0
# iou_threshold = 0.45
# center_variance = 0.1
# size_variance = 0.2
#
# specs = [
#     SSDSpec(10, 16, SSDBoxSizes(60, 105), [2, 3]),
#     SSDSpec(5, 32, SSDBoxSizes(105, 150), [2, 3]),
#     SSDSpec(3, 64, SSDBoxSizes(150, 195), [2, 3]),
#     SSDSpec(2, 120, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(1, 200, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
# ]
#
#
# priors = generate_ssd_priors(specs, image_size)
