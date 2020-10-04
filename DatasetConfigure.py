import os, random
import sys
import click
import json

ERROR_EXIT_CODE = -1

class DatasetConfigure():

    def __init__(self, images_folder, dataset_folder):
        self.images_folder = images_folder
        self.dataset_folder = dataset_folder
        self.validateArguments()

    def validateArguments(self):
        if not os.path.exists(self.images_folder):
            sys.stderr.write(f'{self.images_folder} does not exist\n')
            sys.exit(ERROR_EXIT_CODE)
        if not os.path.exists(self.dataset_folder):
            sys.stderr.write(f'{self.dataset_folder} does not exist\n')
            sys.exit(ERROR_EXIT_CODE)
        self.class_folders = [class_folder for class_folder in os.listdir(os.path.join(self.images_folder, 'validation'))
            if os.path.isdir(os.path.join(os.path.join(self.images_folder, 'validation'), class_folder))]
        if(len(self.class_folders) < 2):
            sys.stderr.write(f'There should be at least two classes (subfolders) in the images folder\n')
            sys.exit(ERROR_EXIT_CODE)

    def write(self, path, data):
        with open(path, "w") as file:
            file.write(data)

    def configure(self):
        for class_folder in self.class_folders:
            for dataset_type in ['train', 'test', 'validation']:
                images_path = os.path.join(os.path.join(self.images_folder, dataset_type), class_folder)
                image_files = [os.path.join(images_path, image_file) for image_file in os.listdir(images_path) \
                    if os.path.isfile(os.path.join(images_path, image_file))]
                self.write(os.path.join(self.dataset_folder, f"{class_folder}_{dataset_type}.json"), json.dumps(image_files))
        self.write(os.path.join(self.dataset_folder, f"labels.json"), json.dumps(self.class_folders))

@click.command()
@click.option('--images_folder', default=f'{os.getcwd()}/images', help='Images folder')
@click.option('--dataset_folder', default=f'{os.getcwd()}/dataset', help='Folder where dataset is stored')
def configureDataset(images_folder, dataset_folder):
    dataset_configure = DatasetConfigure(images_folder, dataset_folder)
    dataset_configure.configure()

if __name__ == '__main__':
    configureDataset()
