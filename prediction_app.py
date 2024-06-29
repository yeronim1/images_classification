import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

model = tf.keras.models.load_model('image_classification_model.keras')
pred_dir = './seg_pred'


def classify_image(model, img_dir, target_size=(150, 150)):
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg') or
                                                                                    fname.endswith('.png')]
    for image_path in img_paths:
        img = Image.open(image_path).resize(target_size)
        img_array = np.array(img).astype('float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        predict_class = classes[np.argmax(predictions)]

        plt.imshow(img)
        plt.title(f'Class: {predict_class}')
        plt.axis('off')
        plt.pause(2)
        plt.show()


classify_image(model, pred_dir)
