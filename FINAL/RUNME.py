import sys
import os
import shutil

# Ensure the parent directories are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import torch
from tqdm import tqdm

from model.Preprocessing.Preprocessing import preprocess_image_from_input
from model.architecture import CNN

from post_processing import format_equation, correct_equation
from compute_equation import compute_equation

class CustomDataset():
    def __init__(self, root_dir, transform=None):
        # print("\n\nInitializing Custom Dataset")
        self.root_dir = root_dir
        # print("Root Directory:", root_dir)
        self.image_paths = []
        
        class_folder = os.path.join(root_dir)
        # print("Class Folder:", class_folder)
        if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    # print("Image Path:", img_path)
                    if img_path.endswith(('.jpg', '.png', '.jpeg')):  # Ensure valid image formats
                        self.image_paths.append(img_path)
                        
                # print("Image Processed")

    def __len__(self):
        return len(self.image_paths)

    def getitem(self):
        """ Get an image and its corresponding label """
        images = []
        self.image_paths = sorted(self.image_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for i in self.image_paths:
            # print("Processing Image Path:", i)
            image = cv2.imread(i, cv2.IMREAD_GRAYSCALE) # Do not convert to RGB
            image = torch.tensor(image).unsqueeze(0)
            images.append(image)
            
        # print("\n\nImages: ", len(images))

        return images


inverted_class_mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "+",
    11: "-",
    12: "forward_slash",
    13: "(",
    14: ")",
    15: "div",
    16: "times",
    17: "x",
    18: "y",
}

def runcode(image_path=''):
   
    # Get the image path from the user
    print('\nEnter image path (CTRL+C to exit):', end='', flush=True)  #  Fix here
    image_path = str(input())  


    # Check if the path is valid
    if not os.path.exists(image_path):
        print('Invalid path.')
        exit()

    # Preprocess the image
    preprocess_image_from_input(image_path)

        
    test_dataset = CustomDataset(root_dir=os.path.join(os.getcwd(), "cache/final_letters"))
    test_dataset_x = test_dataset.getitem()
    test_dataset_x = torch.stack(test_dataset_x).cuda()
    test_dataset_x = test_dataset_x.to(torch.float32).cuda()
    

    output = model(test_dataset_x)
    pred = torch.argmax(output, dim=1)

    pred = [inverted_class_mapping[p.item()] for p in pred]

    print("Predicted Equation: ", end='')
    for p in pred:
        print(p, end='')
    print()

    equation = format_equation(pred)
    equation = correct_equation(equation)
    

    if equation:
        print(f"Corrected Equation: {equation}")
        compute_equation(equation)
    else:
        print("Error: Could not identify a valid equation.")

    # compute_equation(equation)

    cache_files = [
        os.path.join(os.getcwd(), 'cache/cca_output.png'),
        os.path.join(os.getcwd(), 'cache/cropped_image.png'),
        os.path.join(os.getcwd(), 'cache/deskewed_image.png'),
        os.path.join(os.getcwd(), 'cache/image.png'),
    ]

    for file in cache_files:
        if os.path.exists(file):
            os.remove(file)
            # print(f"Deleted: {file}")
        # else:
            # print(f"File not found: {file}")
    

    # Delete the final_letters folder
    cache_folder = os.path.join(os.getcwd(), 'cache/final_letters')

    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
        # print(f"Deleted folder: {cache_folder}")
    # else:
    #     print(f"Folder not found: {cache_folder}")

    print("\nCompleted Execution.")


if __name__ == '__main__':

    # #clear terminal
    # os.system('cls' if os.name == 'nt' else 'clear')

    # # Load the trained model
    # print("Loading model...", flush=True)  # Force print immediately
    # torch.cuda.synchronize()
    # model = CNN().cuda()
    # model.load_state_dict(torch.load("model.pth", map_location="cuda"))

    # model.eval()
    # print("Model loaded successfully.", flush=True)  #  Ensure print shows up
    # runcode()

    #clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    # Load the trained model
    print("Loading model...", flush=True)  #  Force print immediately
    torch.cuda.synchronize()
    num_classes = 157725
    # num_classes = len(inverted_class_mapping)
    model = CNN(num_classes).cuda()
    model.load_state_dict(torch.load("model.pth", map_location="cuda"))

    model.eval()
    print("Model loaded successfully.", flush=True)  #  Ensure print shows up
    runcode()
