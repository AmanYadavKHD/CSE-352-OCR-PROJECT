import cv2
import numpy as np

def smoothen_pixelated_image(image_path, upscale_factor=4):
    """Smoothens a pixelated image by upscaling with bicubic interpolation and applying Gaussian blur."""

    # Load the low-resolution image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Upscale the image using bicubic interpolation
    height, width = img.shape[:2]
    new_size = (width * upscale_factor, height * upscale_factor)
    upscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

    # Apply a slight Gaussian blur to remove pixelation
    smoothed_img = cv2.GaussianBlur(upscaled_img, (5, 5), 0)

    return smoothed_img