#!/usr/bin/python3image_idimage_id
"""Script for creating classes for data preprocessing. It has three classes corresponding to training, testing and prediction.
"""
from dataloaders.transforms import *


class TrainTransform:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            RandomSampleCrop(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None, rng=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels,rng=None):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels,rng)

class RLTrainTransform:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            RandomSampleCrop(),
            # PhotometricDistort(),
            # Expand(self.mean),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None, rng=None: (img / std, boxes, labels),
            ToTensor(),
        ])

        self.augment_no_rnd = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None, rng=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels, rng=None):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        if rng is not None:
            print("RANDOM")
            return self.augment(img, boxes, labels, rng)
        else:
            print("NO RANDOM")
            return self.augment_no_rnd(img, boxes, labels, rng)

class ValTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None,rng=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels,rng=None):
        return self.transform(image, boxes, labels)

class InverseValTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToCV2Image(),
            lambda img, boxes=None, labels=None,rng=None: (img * std, boxes, labels),
            AddMeans(mean),
            # ResizeNotSquare(size),
            # ToAbsoluteCoords(),
            ToInt8C3()
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None, rng=None: (img / std, boxes, labels),
            ToTensor()
        ])
        self.transform_noresize = Compose([
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None,rng=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image, resize=True):
        if resize:
            image, _, _ = self.transform(image)
        else:
            image, _, _ = self.transform_noresize(image)
        return image

class RandTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.mean = mean
        self.size = size

        self.transform = Compose([
            ConvertFromInts(),
            RandomSampleCrop(),
            # PhotometricDistort(),
            # Expand(self.mean),
            RandomMirror(),
            ConvertToInts(),
        ])

    def __call__(self, image, boxes, labels,rng=None):
        return self.transform(image, boxes, labels, rng)

