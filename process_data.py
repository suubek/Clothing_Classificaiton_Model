import csv
import torch
import numpy as np
from os.path import join

class Dataloader(object):
    def __init__(self, path, train_filename, test_filename):
        self.train_filepath = join(path, train_filename)
        self.test_filepath = join(path, test_filename)
    
    def read_images_labels(self, image_filepath):
        labels = []
        images = []

        with open(image_filepath, "r") as f:
            file = csv.reader(f)
            for lines in file:
                labels.append(lines[0])
                image = torch.tensor(lines[1:])
                images.append(image.view(28,28))

        if len(labels) != len(image):
            raise ValueError('Mismatched number of labels and images')

        return images, labels

    def load_data(self):
        train_data, train_labels = self.read_images_labels(self.train_filepath)
        test_data, test_labels = self.read_images_labels(self.test_filepath)
        return (train_data, train_labels), (test_data, test_labels)