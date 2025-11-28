import numpy as np 
from config import Config
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import tensorflow_addons as tfa


def load_numpy_data():
    images_path = os.path.join(Config.Processed_dir, f"{Config.prefix}_image.npy")
    masks_path = os.path.join(Config.Processed_dir, f"{Config.prefix}_masks.npy")

    '''Loading the images and the masks from the processed for the images'''
    images = np.load(images_path)
    masks = np.load(masks_path)

    images = images.astype("float32")/np.max(images) #converting the data type to float 32
    masks = masks.astype("float32")

    return images, masks


def preprocess (image, mask): 
    '''Resize and to make sure a proper dimension to it'''
    image = tf.image.resize(image, [Config.IMG_height, Config.IMG_width])
    mask = tf.image.resize(mask, [Config.IMG_height, Config.IMG_width])

    return image, mask

def augmentation_data(image, mask):
    if Config.augmentation["flip_horizontal"] and tf.random.uniform(()) > 0.5: 
        image = tf.image.flip_left_right(image)
        mask =  tf.image.flip_left_right(mask)
    if Config.augmentation["flip_vertical"] and tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if Config.augmentation["rotation_angle"] > 0: 
        angle = tf.random.uniform((), 
                                  minval=-Config.augmentation["rotation_angle"],
                                  maxval=Config.augmentation["rotation_angle"]) * np.pi/180
        
        image = tfa.image.rotate(image, angle)
        mask = tfa.image.rotate(mask, angle)
    return image, mask

def splitting_data(images, masks):
    X_train, y_train, X_val, y_val = train_test_split(
        images, masks, test_size = Config.validation_split, random_state=42)
    return X_train, y_train, X_val, y_val

def building_dataset(X, y, shuffle=True) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augmentation_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(Config.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def loading_dataset() -> tuple[tf.data.Dataset, tf.data.Dataset]: 
    images, masks = load_numpy_data() #loading the data from the preprocessed data 
    X_train, y_train, X_val, y_val = splitting_data(images, masks)
    train_ds = building_dataset(X_train, y_train)
    val_ds = building_dataset(X_val, y_val)
    return train_ds, val_ds


