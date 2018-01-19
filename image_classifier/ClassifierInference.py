import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import math
import os
import datetime
import cv2
from image_classifier.ImageDataset import ImageDataset
from image_classifier.ClassLabels import ClassLabels

class ClassifierInference():

    def __init__(self, model, class_labels, gpu):
        self.model = model
        self.gpu = gpu
        if torch.cuda.is_available():
            model = model.cuda(self.gpu)
        self.class_labels = class_labels

    def predict(self, image):
        cuda_enabled = torch.cuda.is_available()
        image_as_tensor = ImageDataset.preprocess(image)
        image_as_tensor = image_as_tensor.unsqueeze(0)
        if cuda_enabled:
            image_as_tensor = image_as_tensor.cuda(self.gpu)
        predictions = self.model.forward(Variable(image_as_tensor))
        predictions_probabilities = torch.nn.functional.sigmoid(predictions).cpu().data
        _, class_predictions = torch.topk(predictions_probabilities, 1, 0)
        class_index = int(class_predictions.numpy()[0])
        class_label = self.class_labels[class_index]
        probability = predictions_probabilities[class_index]
        return class_label, probability
