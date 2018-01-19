import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import math
import os
import datetime
import click
from PretrainedResnet import PretrainedResnet
from ImageDataset import ImageDataset
from F1Score import F1Score
from ClassLabels import ClassLabels

class ClassifierTrain():

    def __init__(self, model_file, class_labels, images_folder, dataset_folder, model, num_epochs, batch_size, learning_rate, gpu, visual_logging=False):
        self.model = model
        self.images_folder = images_folder
        self.dataset_folder =dataset_folder
        self.visual_logging = visual_logging
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gpu = gpu
        self.model_file = model_file
        self.class_labels = class_labels

    def log(text):
        print(f"{datetime.datetime.utcnow()} -- {text}")

    def testF1Score(self):
        f1_score = F1Score(len(self.class_labels))
        dataset_test = ImageDataset(self.images_folder, self.dataset_folder, False, self.visual_logging)
        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=4)
        cuda_enabled = torch.cuda.is_available()
        for i, (targets, images) in enumerate(test_loader):
            sys.stdout.flush()
            if cuda_enabled:
                targets = targets.cuda(self.gpu)
                images = images.cuda(self.gpu)
            targets = Variable(targets)
            images = Variable(images)
            predictions = self.model.forward(images)
            f1_score.add(predictions.data.cpu(), targets.data.cpu())
        return f1_score.compute()

    def testAndSaveIfImproved(self):
        average_current_test_f1_score = self.testF1Score()
        if average_current_test_f1_score > self.best_test_f1_score:
            ClassifierTrain.log(f"Model Improved. Previous Best Test F1-Score {self.best_test_f1_score:{1}.{4}} | Current Best Test F1-Score  {average_current_test_f1_score:{1}.{4}}")
            self.best_test_f1_score = average_current_test_f1_score
            ClassifierTrain.log(f"... saving model ...")
            torch.save(self.model.float(), self.model_file)
            ClassifierTrain.log(f"... model saved.")
        else:
            ClassifierTrain.log(f"Model did *NOT* Improve. Current Best Test F1-Score {self.best_test_f1_score:{1}.{4}} | Current Test F1-Score {average_current_test_f1_score:{1}.{4}}")

    def train(self):
        criterion = nn.CrossEntropyLoss()
        cuda_enabled = torch.cuda.is_available()
        if cuda_enabled:
            criterion = criterion.cuda(self.gpu)
            self.model =  self.model.cuda(self.gpu)
        optimizer = torch.optim.SGD([
                    {'params': self.model.parameters()}
                ], lr=self.learning_rate)
        self.best_test_f1_score = self.testF1Score()
        ClassifierTrain.log(f"Initial Test F1-Score (top1) {self.best_test_f1_score:{1}.{4}} ")
        for epoch in range(self.num_epochs):
            f1_score_epoch = F1Score(len(self.class_labels))
            ClassifierTrain.log(f"Epoch {epoch}")
            dataset_train = ImageDataset(self.images_folder, self.dataset_folder, True, self.visual_logging)
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=4)
            for i, (targets, images) in enumerate(train_loader):
                f1_score_batch = F1Score(len(self.class_labels))
                sys.stdout.flush()
                if cuda_enabled:
                    targets = targets.cuda(self.gpu)
                    images = images.cuda(self.gpu)
                targets = Variable(targets)
                images = Variable(images)
                predictions = self.model.forward(images)
                f1_score_batch.add(predictions.data.cpu(), targets.data.cpu())
                f1_score_epoch.add(predictions.data.cpu(), targets.data.cpu())
                optimizer.zero_grad()
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                mean_loss = loss.data[0]
                ClassifierTrain.log(f'Epoch [{epoch+1}/{self.num_epochs}] -- Iter [{i+1}/{math.ceil(len(dataset_train)/self.batch_size)}] -- Train Loss: {mean_loss:{1}.{4}} -- Train F1-Score in Batch: {f1_score_batch.compute():{1}.{4}} -- Train F1-Score in Epoch: {f1_score_epoch.compute():{1}.{4}}')
            self.testAndSaveIfImproved()
