import os
import time
import argparse

import imageio
from PIL import Image
import numpy as np
import torch
import torch.optim as optim

from utility import load_and_process_img, deprocess_img, create_timedir, load_config
from model import get_style_loss, get_content_loss, gram_matrix, VGG19

config = load_config("./config.yml")

NUM_ITER = config["num_iterations"]
CONTENT_WEIGHT = config["content_weight"]
STYLE_WEIGHT = config["style_weight"]
LR = config["learning_rate"]


def get_feature_representations(model, content_path, style_path, device):
    """
    Helper function to compute our content and style feature representations.
    """
    # Load our images in
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # batch compute content and style features
    style_outputs, _ = model(style_image.to(device))
    _, content_outputs = model(content_image.to(device))

    return style_outputs, content_outputs


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """
    This function will compute the loss total loss.
    """
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    style_output_features, content_output_features = model(init_image)

    num_content_layers = len(content_output_features)
    num_style_layers = len(style_output_features)

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * \
            get_style_loss(comb_style, target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * \
            get_content_loss(comb_content, target_content)

    style_score *= style_weight
    content_score *= content_weight
    total_loss = style_score + content_score
    return total_loss, style_score, content_score


def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1,
                       style_weight=1e6,
                       learning_rate=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG19().to(device).eval()

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(
        model, content_path, style_path, device)
    gram_style_features = [gram_matrix(style_feature)
                           for style_feature in style_features]

    # load image
    init_image = load_and_process_img(content_path).to(device)
    init_image = init_image.requires_grad_(True)

    optimizer = optim.Adam([init_image], lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-01)

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # For displaying
    display_interval = num_iterations/10
    start_time = time.time()
    global_start = time.time()

    savepath = create_timedir("./data/output")
    for i in range(num_iterations):
        start_time = time.time()
        init_image.data.clamp_(0, 1)
        optimizer.zero_grad()
        loss, style_score, content_score = compute_loss(**cfg)
        loss.backward()
        optimizer.step()

        if (i+1) % display_interval == 0:
            plot_img = init_image.detach().cpu()
            plot_img = deprocess_img(plot_img)

            filename = f"iter-{i+1}.png"
            imageio.imwrite(os.path.join(
                savepath, filename), plot_img)
            print(f'[INFO] saved {filename} to {savepath}')
            print('Iteration: {}'.format(i+1))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss.item(), style_score.item(), content_score.item(), time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    final_img = deprocess_img(init_image.detach().cpu())
    return final_img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="path to input image",
                    default="./data/Green_Sea_Turtle_grazing_seagrass.jpg")
    ap.add_argument("-s", "--style", help="path to image whose style is to be imitated",
                    default="data/The_Great_Wave_off_Kanagawa.jpg")
    args = vars(ap.parse_args())

    run_style_transfer(args["input"], 
                       args["style"],
                       num_iterations=NUM_ITER,
                       content_weight=CONTENT_WEIGHT, 
                       style_weight=STYLE_WEIGHT, 
                       learning_rate=LR)
