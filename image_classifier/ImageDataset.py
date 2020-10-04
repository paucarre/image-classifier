import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os, random, string
from torch.autograd import Variable
import sys
import numpy as np
import cv2
import json
import click
import torch
import traceback
from torchvision import transforms, utils
from image_classifier.ClassLabels import ClassLabels

class ImageDataset(Dataset):

    def __init__(self, images_folder, dataset_folder, dataset_type, visual_logging=False, seed=None, is_tiny=False):
        if is_tiny:
            self.width=32
            self.height=32
        else:
            self.width=244
            self.height=244
        random.seed(seed)
        self.dataset_folder = dataset_folder
        self.images_folder = images_folder
        self.is_train = dataset_type == 'train'
        self.visual_logging = visual_logging
        self.dataset_type = dataset_type
        self.dataset = None
        if self.is_train:
            self.dataset = {}
        else:
            self.dataset = []
        class_labels = ClassLabels(self.dataset_folder)
        self.labels = class_labels.labels
        self.length = 0
        for label_index, label in enumerate(self.labels):
            images_path = os.path.join(self.dataset_folder, f"{label}_{dataset_type}.json")
            with open(images_path, "r") as images_file:
                images_in_labels = json.load(images_file)
                if self.is_train: # per-class image for uniform sampling
                    random.shuffle(images_in_labels)
                    self.dataset[label_index] = images_in_labels
                else:
                    images_with_labels = [(label_index, images_in_label) for images_in_label in images_in_labels]
                    self.dataset = self.dataset + images_with_labels
                self.length = self.length + len(images_in_labels)

    def preprocess(image, width, height):
        image = ImageDataset.scale(image, width, height)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        return preprocess(image)

    def __len__(self):
        return self.length

    def cropImage(self, image, y_start, y_end, x_start, x_end):
        crop = np.ones( (y_end - y_start, x_end - x_start, 3), dtype=np.uint8)
        for channel in range(0, 3):
            crop[:, :, channel] = image[y_start:y_end, x_start:x_end, channel]
        return crop

    def randomIntensity(self, image):
        luminosity_factor = random.uniform(0.5, 1.5)
        for channel in range(0, 3):
            image[:, :, channel] = (((image[:, :, channel] * luminosity_factor) > 255) * image[:, :, channel]) + (((image[:, :, channel] * luminosity_factor) <= 255) * image[:, :, channel] * luminosity_factor)
        return image

    def centerCropParameters(self, image):
        height = image.shape[0]
        width  = image.shape[1]
        crop_percentage = 0.1
        y_start = int(height * crop_percentage)
        y_end = height - int(height * crop_percentage)
        x_start = int(width * crop_percentage)
        x_end = width - int(width * crop_percentage)
        return y_start, y_end, x_start, x_end

    def randomCropParameters(self, image):
        height = image.shape[0]
        width  = image.shape[1]
        crop_percentage_max = 0.2
        y_start = int(random.uniform(0, int(height * crop_percentage_max)))
        y_end = height - int(random.uniform(0, int(height * crop_percentage_max)))
        x_start = int(random.uniform(0, int(width * crop_percentage_max)))
        x_end = width - int(random.uniform(0, int(width * crop_percentage_max)))
        return y_start, y_end, x_start, x_end

    def randomRotation(self, image):
        random_angle_distorsion =  (-random.random() * 20) +  10
        image_height, image_width, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D( (int(image_width / 2), int(image_height / 2) ), random_angle_distorsion, 1)
        image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height))
        return image

    def scale(image, width, height):
        return cv2.resize(image, (width, height))

    def __getitem__(self, index):
        attempts = 0
        while(attempts < 10):
            try:
                image_path = None
                label_index = None
                if self.is_train:
                    label_index = random.randint(0, len(self.labels) - 1)
                    image_path = self.dataset[label_index][index % len(self.dataset[label_index])]
                else:
                    label_index, image_path = self.dataset[index % len(self.dataset)]
                image_path = os.path.join(self.images_folder, image_path)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if self.is_train:
                    if self.visual_logging:
                        cv2.imshow(f'original_image', image)
                    if random.choice([True, False]):
                        image = self.randomRotation(image)
                    if self.visual_logging:
                        cv2.imshow(f'random_rotation', image)
                    if random.choice([True, False]):
                        y_start, y_end, x_start, x_end = self.randomCropParameters(image)
                        image = self.cropImage(image,  y_start, y_end, x_start, x_end)
                    if self.visual_logging:
                        cv2.imshow(f'random_crop', image)
                    if random.choice([True, False]):
                        image = self.randomIntensity(image)
                    if self.visual_logging:
                        cv2.imshow(f'random_intensity', image)
                        cv2.waitKey(0)
                return label_index, ImageDataset.preprocess(image, self.width, self.height)
            except BaseException as e:
                sys.stderr.write(f'Unable to load image {image_path}. Skipping\n')
                sys.stderr.write(traceback.format_exc())
            index = (index + 1) % len(self)
            attempts = attempts + 1
        sys.stderr.write(f'Major error loading images from dataset.\n')
        return None
