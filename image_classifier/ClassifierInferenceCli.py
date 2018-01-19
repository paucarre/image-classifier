import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import math
import os
import datetime
import click
import cv2
from ImageDataset import ImageDataset
from ClassLabels import ClassLabels
from ClassifierInference import ClassifierInference

@click.command()
@click.option('--model_file', default=f'{os.getcwd()}/models/model_best.model', help='Path of the filename where the model is saved.')
@click.option('--image_path', default='', help='Image path used to predict its class')
@click.option('--dataset_folder', default=f'{os.getcwd()}/dataset', help='Folder where dataset is stored')
@click.option('--gpu', default=0, help='Only used if CUDA is detected. GPU index. Index starts from 0 to N - 1 for N GPUs in your system.')
def inference(model_file, image_path, dataset_folder, gpu):
    class_labels = ClassLabels(dataset_folder).labels
    model = torch.load(model_file)
    classifier_inference = ClassifierInference(model, class_labels, gpu)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    class_label, probability = classifier_inference.predict(image)
    print("{ " + f"'class_label': '{class_label}', 'probability': {probability:{1}.{4}}" + " }")

if __name__ == '__main__':
    inference()
