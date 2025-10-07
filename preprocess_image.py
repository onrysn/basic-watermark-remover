import numpy as np
import cv2
from PIL import Image

def preprocess_image(image, watermark_type):
    image_type = ''
    preprocessed_mask_image = np.array([])

    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    image_h, image_w = image.shape[:2]
    print("image size: {}".format(image.shape))

    if image_w >= image_h:
        image_type = "landscape"
    else:
        image_type = "portrait"

    mask_path = f"utils/{watermark_type}/{image_type}/mask.png"
    mask_image = Image.open(mask_path)
    if mask_image.mode != "RGB":
        mask_image = mask_image.convert("RGB")
    mask_image = np.array(mask_image)
    print("mask image size: {}".format(mask_image.shape))

    preprocessed_mask_image = cv2.resize(mask_image, (image_w, image_h))
    print("resized mask shape:", preprocessed_mask_image.shape)

    if preprocessed_mask_image.size != 0:
        if image.shape != preprocessed_mask_image.shape:
            raise ValueError("Image and mask dimensions do not match.")
        grid = 8
        image = image[:image_h//grid*grid, :image_w//grid*grid, :]
        preprocessed_mask_image = preprocessed_mask_image[:image_h//grid*grid, :image_w//grid*grid, :]
        image = np.expand_dims(image, 0)
        preprocessed_mask_image = np.expand_dims(preprocessed_mask_image, 0)
        input_image = np.concatenate([image, preprocessed_mask_image], axis=2)
        return input_image
    else:
        return preprocessed_mask_image
