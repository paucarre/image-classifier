import os
import click
import torch
from image_classifier.PretrainedResnet import PretrainedResnet
from image_classifier.TinyPretrainedResnet import TinyPretrainedResnet
from image_classifier.ClassLabels import ClassLabels
from image_classifier.ClassifierTrain import ClassifierTrain

def loadModel(best_model_file, is_tiny, reset_model, class_labels):
    model = None
    if reset_model:
        if is_tiny:
            model = TinyPretrainedResnet(len(class_labels))
        else:
            model = PretrainedResnet(len(class_labels))
    else:
        model = torch.load(best_model_file)
    return model

@click.command()
@click.option('--model_file', default=f'{os.getcwd()}/models/model_best.model', help='Path of the filename where the model is saved.')
@click.option('--max_epochs_test_not_improving', default=20, help='Maximum number of epochs without test improving')
@click.option('--is_tiny', default=True, help='Whether input images are small at 32x32')
@click.option('--images_folder', default=f'{os.getcwd()}/images', help='Folder where the images are stored. Each subfolder shall contain the class label and each subfolder has to contain all the images')
@click.option('--dataset_folder', default=f'{os.getcwd()}/dataset', help='Folder where dataset is stored')
@click.option('--visual_logging', default=False, help='Only Desktop. Display additional logging using images (e.g. image sampling). Do not use it in a server, it requires a desktop environment.')
@click.option('--reset_model', default=False, help='Reset model (start model from scratch).')
@click.option('--num_epochs', default=1000, help='Number of epochs.')
@click.option('--batch_size', default=32, help='Batch size.')
@click.option('--learning_rate', default= 0.0001, help='Learning rate')
@click.option('--gpu', default=0, help='Only used if CUDA is detected. GPU index. Index starts from 0 to N - 1 for N GPUs in your system.')
def train(model_file, max_epochs_test_not_improving, is_tiny, images_folder, dataset_folder, visual_logging, reset_model, num_epochs, batch_size, learning_rate, gpu):
    class_labels = ClassLabels(dataset_folder).labels
    model = loadModel(model_file, is_tiny, reset_model, class_labels)
    classifier_train = ClassifierTrain(model_file, max_epochs_test_not_improving, is_tiny, class_labels, images_folder, dataset_folder, model, num_epochs, batch_size, learning_rate, gpu, visual_logging)
    classifier_train.train()

if __name__ == '__main__':
    train()
