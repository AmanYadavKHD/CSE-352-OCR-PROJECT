import os
import sys
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import argparse
from model.architecture import CNN
# Ensure the parent directories are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Check device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# class_mapping = {
#     "0": 0,
#     "1": 1,
#     "2": 2,
#     "3": 3,
#     "4": 4,
#     "5": 5,
#     "6": 6,
#     "7": 7,
#     "8": 8,
#     "9": 9,
#     "A": 10,
#     "B": 11,
#     "C": 12,
#     "D": 13,
#     "E": 14,
#     "F": 15,
#     "G": 16,
#     "H": 17,
#     "I": 18,
#     "J": 19,
#     "K": 20,
#     "L": 21,
#     "M": 22,
#     "N": 23,
#     "O": 24,
#     "P": 25,
#     "Q": 26,
#     "R": 27,
#     "S": 28,
#     "T": 29,
#     "U": 30,
#     "V": 31,
#     "W": 32,
#     "X": 33,
#     "Y": 34,
#     "Z": 35,
#     "+": 36,
#     "-": 37,
#     "forward_slash": 38,
# }

class_mapping = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "plus": 10,
    "minus": 11,
    "forward_slash": 12,
    # "dot":13,
    "left_bracket":13,
    "right_bracket":14,
    "div":15,
    "times":16,
    "x":17,
    "y":18,
}


# Define Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # print("Initializing Custom Dataset")
        self.root_dir = root_dir
        self.transform = transform

        # Get all subfolders (class labels)
        self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]

        self.class_to_idx = {cls_name: class_mapping[cls_name] for cls_name in self.classes if cls_name in class_mapping}

        # Initialize list of image paths and corresponding labels
        self.image_paths = []
        self.labels = []

        # Loop through all subfolders and images
        for cls_name in self.classes:
            class_folder = os.path.join(root_dir, "forward_slash" if cls_name == "/" else cls_name)
            # print("Class Folder:", class_folder)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    # print("Image Path:", img_path)
                    if img_path.endswith(('.jpg', '.png', '.jpeg')):  # Ensure valid image formats
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])
                # print("Image Processed")

    def __len__(self):
        return len(self.image_paths)

    def getitem(self):
        """ Get an image and its corresponding label """
        images = []
        labels = []

        for i, l in zip(self.image_paths, self.labels):
            # print("Processing Image Path:", i)
            image = cv2.imread(i, cv2.IMREAD_GRAYSCALE) # Do not convert to RGB
            image = torch.tensor(image).unsqueeze(0)
            images.append(image)
            labels.append(l)
        return images, labels


# Set the paths to your training and testing dataset directories
train_dir = os.path.join(os.getcwd(), "data\\split_dataset\\train")  # Training data directory
test_dir = os.path.join(os.getcwd(), "data\\split_dataset\\test")  # Testing data directory

# # Load the datasets (as a whole)
# print("Loading Training Dataset")
# train_dataset = CustomDataset(root_dir=train_dir)
# print("Loading Testing Dataset")
# test_dataset = CustomDataset(root_dir=test_dir)
# print("Datasets Loaded")

# # Get images and labels
# train_dataset_x, train_dataset_y = train_dataset.getitem()
# test_dataset_x, test_dataset_y = test_dataset.getitem()

# print("Converting to Tensors")

# train_dataset_x = np.array(train_dataset_x)
# train_dataset_y = np.array(train_dataset_y)
# test_dataset_x = np.array(test_dataset_x)
# test_dataset_y = np.array(test_dataset_y)

# print('Train images = ', len(train_dataset_y))
# print('Test images = ', len(test_dataset_y))

# os.makedirs('cache', exist_ok=True)

# np.save("cache/train_dataset_x.npy", train_dataset_x)
# np.save("cache/train_dataset_y.npy", train_dataset_y)
# np.save("cache/test_dataset_x.npy", test_dataset_x)
# np.save("cache/test_dataset_y.npy", test_dataset_y)

# print('Datasets Saved')

import argparse

from model.architecture import get_CNN
model = get_CNN(num_classes=157725)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
args = parser.parse_args()

if args.mode == "train":
    print("Loading Training Dataset")
    train_dataset = CustomDataset(root_dir=train_dir)
    test_dataset = CustomDataset(root_dir=test_dir)

    # Load dataset ONLY for training
    train_dataset_x, train_dataset_y = train_dataset.getitem()
    test_dataset_x, test_dataset_y = test_dataset.getitem()

    print("Converting to Tensors")
    train_dataset_x = np.array(train_dataset_x)
    train_dataset_y = np.array(train_dataset_y)
    test_dataset_x = np.array(test_dataset_x)
    test_dataset_y = np.array(test_dataset_y)

    print('Train images = ', len(train_dataset_y))
    print('Test images = ', len(test_dataset_y))

    os.makedirs('cache', exist_ok=True)
    np.save("cache/train_dataset_x.npy", train_dataset_x)
    np.save("cache/train_dataset_y.npy", train_dataset_y)
    np.save("cache/test_dataset_x.npy", test_dataset_x)
    np.save("cache/test_dataset_y.npy", test_dataset_y)

    print('Datasets Saved')

elif args.mode == "test":
    print("Running Inference Mode")
    num_classes = 157725
    model = CNN(num_classes).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    # Inference Example (NO dataset loading)
    test_image = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)
    test_image = torch.tensor(test_image).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        prediction = model(test_image)

    print("Predicted Output:", prediction)
