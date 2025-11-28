import numpy as np
import nibabel as nib
import tensorflow as tf
from config import Config
import os


#first defying the loading of the volumetric data 
def loading_raw_data (filepath : str, raw_data: np.ndarray) -> np.ndarray:
    image = nib.load(filepath) #loading the data
    image = image.get_fdata() #converting it to acceptable numpy 
    image = (image - np.mean(image))/(np.std(image) + 1e-10) #adding a smoothening factor for normalization 
    return image

#loadin the volume as 2d slices - which are better for the model : 

def volume_slices(volume):
    slices = []
    for i in range(volume.shape[2]): #going along the 3d slicing
        slice = volume [ : , : , i] #first slice 
        slice = tf.image.resize(slice[..., np.newaxis], (Config.IMG_height, Config.IMG_width))
        slices.append(slice)
    slices = np.stack(slices)
    return slices #return the slices that are in a list

#adding the augmentation part to add the to the raw normalized data
def augmenting_data (image: np.ndarray, mask: np.ndarray): 
    if Config.augmentation["flip_horizontal"] and tf.random.uniform(()) > 0.5: 
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if Config.augmentation["flip_vertical"] and tf.random.uniform(()) > 0.5: 
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if Config.augmentation["rotation_angle"] > 0:
        angle = tf.random.uniform((), 
                                minval = -Config.augmentation["rotation_angle"],
                                maxval = Config.augmentation["rotation_angle"])* np.pi/180
        image  = tfa.image.rotate(image, angle)
        mask  = tfa.image.rotate(mask, angle)
    return image, mask

def building_dataset(images, masks): 
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    if Config.shuffle: 
        dataset = dataset.shuffle(buffer_size= len(images)) #shuffling it 
    dataset = dataset.map(lambda x, y: augmenting_data(x, y), num_parallel_calls= tf.data.AUTOTUNE) #mapping all the elements
    dataset = dataset.batch(batch_size=Config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) #fetching another while the current one is handled by the GPU 
    return dataset

def saving_dataset(images, masks, prefix = Config.prefix):
    np.save(os.path.join(Config.Processed_dir, f"{prefix}_image.npy"), images)
    np.save(os.path.join(Config.Processed_dir, f"{prefix}_masks.npy"), masks)

            

