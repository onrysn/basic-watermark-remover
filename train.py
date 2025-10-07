import os
import cv2
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Eğitim ayarları
BATCH_SIZE = 4
IMG_SHAPE = (1000, 1000)
MAX_ITERS = 10000
CHECKPOINT_DIR = 'model_onur'
FLIST_PATH = 'data/fmgproducts/train_shuffled.flist'
MASK_DIR = 'data/fmgproducts/mask/'

# Basit model (encoder-decoder)
def build_model(input_tensor):
    with tf.compat.v1.variable_scope('inpaint_net'):
        x = tf.compat.v1.layers.conv2d(input_tensor, 64, 5, activation=tf.nn.relu, padding='SAME')
        x = tf.compat.v1.layers.conv2d(x, 128, 3, activation=tf.nn.relu, padding='SAME')
        x = tf.compat.v1.layers.conv2d(x, 64, 3, activation=tf.nn.relu, padding='SAME')
        x = tf.compat.v1.layers.conv2d(x, 3, 3, activation=tf.nn.tanh, padding='SAME')
        return x

# Veri yükleyici
def load_batch(flist_path, mask_dir, batch_size):
    with open(flist_path, 'r') as f:
        image_paths = f.read().splitlines()

    while True:
        np.random.shuffle(image_paths)
        for i in range(0, len(image_paths), batch_size):
            batch_images = []
            batch_masks = []
            for img_path in image_paths[i:i+batch_size]:
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMG_SHAPE)
                img = img.astype(np.float32) / 127.5 - 1.0

                mask_name = os.path.basename(img_path).replace('.jpg', '_mask.png')
                mask_path = os.path.join(mask_dir, mask_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, IMG_SHAPE)
                mask = (mask > 127).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)

                batch_images.append(img)
                batch_masks.append(mask)

            batch_images = np.array(batch_images)
            batch_masks = np.array(batch_masks)
            yield batch_images, batch_masks

# Eğitim döngüsü
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    image_ph = tf.compat.v1.placeholder(tf.float32, [None, IMG_SHAPE[0], IMG_SHAPE[1], 3])
    mask_ph = tf.compat.v1.placeholder(tf.float32, [None, IMG_SHAPE[0], IMG_SHAPE[1], 1])

    incomplete = image_ph * (1.0 - mask_ph)
    input_tensor = tf.concat([incomplete, mask_ph], axis=3)
    output = build_model(input_tensor)
    complete = output * mask_ph + incomplete * (1.0 - mask_ph)

    loss = tf.reduce_mean(tf.abs(image_ph - complete))
    train_op = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.compat.v1.train.Saver()
    data_gen = load_batch(FLIST_PATH, MASK_DIR, BATCH_SIZE)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(MAX_ITERS):
            batch_images, batch_masks = next(data_gen)
            feed_dict = {image_ph: batch_images, mask_ph: batch_masks}
            _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)

            if step % 100 == 0:
                print(f"🧠 Step {step} | Loss: {loss_val:.4f}")

            if step % 1000 == 0 or step == MAX_ITERS - 1:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f'model.ckpt-{step}')
                saver.save(sess, ckpt_path)
                print(f"💾 Checkpoint kaydedildi: {ckpt_path}")

if __name__ == "__main__":
    train()
