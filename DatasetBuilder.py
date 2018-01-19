import os, random
import sys
import click
import json

ERROR_EXIT_CODE = -1

class DatasetBuilder():

    def validateArguments(self):
        if not os.path.exists(self.images_folder):
            sys.stderr.write(f'{self.images_folder} does not exist\n')
            sys.exit(ERROR_EXIT_CODE)
        if not os.path.exists(self.dataset_folder):
            sys.stderr.write(f'{self.dataset_folder} does not exist\n')
            sys.exit(ERROR_EXIT_CODE)
        if 0.0 > self.test_percentage or self.test_percentage > 100.0:
            sys.stderr.write(f'Test percentage {self.test_percentage} should be between 0 and 100\n')
            sys.exit(ERROR_EXIT_CODE)
        self.class_folders = [class_folder for class_folder in os.listdir(self.images_folder) if os.path.isdir(os.path.join(self.images_folder, class_folder))]
        if(len(self.class_folders) < 2):
            sys.stderr.write(f'There should be at least two classes (subfolders) in the images folder\n')
            sys.exit(ERROR_EXIT_CODE)

    def __init__(self, images_folder, test_percentage, dataset_folder):
        self.images_folder = images_folder
        self.test_percentage = test_percentage
        self.dataset_folder = dataset_folder
        self.validateArguments()

    def write(self, path, data):
        with open(path, "w") as file:
            file.write(data)

    def build(self):
        for class_folder in self.class_folders:
            class_path = os.path.join(self.images_folder, class_folder)
            image_files = [os.path.join(class_folder, image_file) for image_file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, image_file))]
            last_test_index = int(len(image_files) * self.test_percentage / 100.0)
            random.shuffle(image_files)
            test_set = image_files[0:last_test_index]
            train_set = image_files[last_test_index + 1:]
            self.write(os.path.join(self.dataset_folder, f"{class_folder}_test.json"), json.dumps(test_set))
            self.write(os.path.join(self.dataset_folder, f"{class_folder}_train.json"), json.dumps(train_set))
        self.write(os.path.join(self.dataset_folder, f"labels.json"), json.dumps(self.class_folders))

@click.command()
@click.option('--images_folder', default=f'{os.getcwd()}/images', help='Images folder')
@click.option('--test_percentage', default=10.0, help='Percentage of test set between 0 and 100')
@click.option('--dataset_folder', default=f'{os.getcwd()}/dataset', help='Folder where dataset is stored')
def buildDataset(images_folder, test_percentage, dataset_folder):
    dataset_builder = DatasetBuilder(images_folder, test_percentage, dataset_folder)
    dataset_builder.build()

if __name__ == '__main__':
    buildDataset()
