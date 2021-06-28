import os
from datetime import datetime
import yaml

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def load_img(path_to_img, max_dim=512):
    """
    load image using path, resize to max_dim and return as numpy array
    """ 
    img = Image.open(path_to_img)
    long_side = max(img.size)
    scale = max_dim/long_side
    img = img.resize(
        (round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = np.array(img)
    return img


def imshow(img, title=None):
    """
    plot image using pyplot
    """
    # Normalize for display
    out = img.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(
    mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std = [1/0.229, 1/0.224, 1/0.225]
)

def load_and_process_img(path_to_img, normalized=False):
    """
    load image, convert to tensor (optionally, normalize) and return
    with batch dimension
    """
    img = load_img(path_to_img)
    img = transforms.ToTensor()(img)
    if normalized:
        img = normalize(img)
    return torch.unsqueeze(img, 0) # returns a tensor (b, c, h, w)

def deprocess_img(processed_img, normalized=False):
    """
    convert from tensor with batch dimension to a numpy array
    (optionally, denormalize)
    """
    x = torch.squeeze(processed_img)
    if normalized:
        x = inv_normalize(x)
    x = x.numpy().transpose((1, 2, 0))
    x = (np.clip(x, 0, 1) * 255).astype('uint8')
    return x # return numpy array (h, w, c)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_timedir(path="./"):
    now = datetime.now()
    dirname = now.strftime("%d-%m-%Y-%H-%M-%S")
    full_path = os.path.join(path, dirname)
    os.makedirs(full_path)
    return full_path

def load_config(config_name, CONFIG_PATH="./"):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    content_path = "../data/Green_Sea_Turtle_grazing_seagrass.jpg"
    style_path = "../data/The_Great_Wave_off_Kanagawa.jpg"
    plt.figure(figsize=(10, 10))

    content = load_img(content_path).astype('uint8')
    style = load_img(style_path).astype('uint8')

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')
    plt.show()
