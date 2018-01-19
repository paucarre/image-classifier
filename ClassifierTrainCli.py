import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import math
import os
import datetime
import click
from image_classifier.PretrainedResnet import PretrainedResnet
from image_classifier.ImageDataset import ImageDataset
from image_classifier.F1Score import F1Score
from image_classifier.ClassLabels import ClassLabels
from image_classifier.ClassifierTrain import ClassifierTrain

def loadModel(best_model_file, reset_model, class_labels):
    model = None
    if reset_model:
        model = PretrainedResnet(len(class_labels))
    else:
        model = torch.load(best_model_file)
    ClassifierTrain.log(model)
    return model

@click.command()
@click.option('--model_file', default=f'{os.getcwd()}/models/model_best.model', help='Path of the filename where the model is saved.')
@click.option('--images_folder', default=f'{os.getcwd()}/images', help='Folder where the images are stored. Each subfolder shall contain the class label and each subfolder has to contain all the images')
@click.option('--dataset_folder', default=f'{os.getcwd()}/dataset', help='Folder where dataset is stored')
@click.option('--visual_logging', default=False, help='Only Desktop. Display additional logging using images (e.g. image sampling). Do not use it in a server, it requires a desktop environment.')
@click.option('--reset_model', default=False, help='Reset model (start model from scratch).')
@click.option('--num_epochs', default=10000, help='Number of epochs.')
@click.option('--batch_size', default=32, help='Batch size.')
@click.option('--learning_rate', default= 0.0001, help='Learning rate')
@click.option('--gpu', default=0, help='Only used if CUDA is detected. GPU index. Index starts from 0 to N - 1 for N GPUs in your system.')
def train(model_file, images_folder, dataset_folder, visual_logging, reset_model, num_epochs, batch_size, learning_rate, gpu):
    class_labels = ClassLabels(dataset_folder).labels
    model = loadModel(model_file, reset_model, class_labels)
    classifier_train = ClassifierTrain(model_file, class_labels, images_folder, dataset_folder, model, num_epochs, batch_size, learning_rate, gpu, visual_logging)
    classifier_train.train()

if __name__ == '__main__':
    train()
