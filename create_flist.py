import os, random

image_dir = 'data/fmgproducts/train'
flist_path = 'data/fmgproducts/train_shuffled.flist'

files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(files)

with open(flist_path, 'w') as f:
    for file in files:
        f.write(file + '\n')

print("✅ train_shuffled.flist dosyası oluşturuldu.")
