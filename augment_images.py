import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# 📁 Girdi klasörleri
INPUT_DIR = 'test_images'
MASK_PATH = 'utils/fmgproducts/landscape/mask.png'

# 📁 Çıktı klasörleri
OUTPUT_IMG_DIR = 'augmented_images'
OUTPUT_MASK_DIR = os.path.join(OUTPUT_IMG_DIR, 'masks')
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# 🔁 Varyasyon fonksiyonları
def rotate(img, angle):
    return img.rotate(angle, expand=True)

def flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def brightness(img, factor):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def add_noise(img):
    arr = np.array(img)
    noise = np.random.normal(0, 10, arr.shape).astype(np.uint8)
    noisy = cv2.add(arr, noise)
    return Image.fromarray(noisy)

def blur(img):
    arr = np.array(img)
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)
    return Image.fromarray(blurred)

def zoom(img, scale=1.1):
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return img.crop((left, top, left + w, top + h))

# 🔄 Tüm varyasyonları uygula (görsel ve maske birlikte)
def apply_all_variations(img, mask):
    variations = []
    variations.append((img, mask))  # Orijinal
    variations.append((rotate(img, 10), rotate(mask, 10)))
    variations.append((rotate(img, -10), rotate(mask, -10)))
    variations.append((flip(img), flip(mask)))
    variations.append((brightness(img, 1.2), mask))
    variations.append((brightness(img, 0.8), mask))
    variations.append((contrast(img, 1.3), mask))
    variations.append((contrast(img, 0.7), mask))
    variations.append((add_noise(img), mask))
    variations.append((blur(img), mask))
    variations.append((zoom(img, 1.1), zoom(mask, 1.1)))
    return variations

# 🚀 Ana döngü
counter = 1
mask_base = Image.open(MASK_PATH).convert("L")

for fname in os.listdir(INPUT_DIR):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(INPUT_DIR, fname)
        img = Image.open(path).convert("RGB")
        variations = apply_all_variations(img, mask_base)
        for i, (var_img, var_mask) in enumerate(variations):
            img_name = f"img{counter:04d}_v{i}.jpg"
            mask_name = f"img{counter:04d}_v{i}_mask.png"
            var_img.save(os.path.join(OUTPUT_IMG_DIR, img_name))
            var_mask.save(os.path.join(OUTPUT_MASK_DIR, mask_name))
        counter += 1

print(f"✅ {counter - 1} görsel için varyasyonlar ve maskeler üretildi.")
