import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models
from config import Config


#developing a comvolution block that will be reused in the u-net architecture
"""
Lets see what is the usually the convolution block doing
"""
def conv_block(inputs, num_filters, name_prefix): 
    x = layers.Conv2D(num_filters, (3,3), padding = "same", name = f"{name_prefix}_conv1")(inputs) #input layer
    x = layers.BatchNormalization(name = f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name = f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(num_filters, (3,3), padding = "same", name = f"{name_prefix}_conv2")(x) #input layer
    x = layers.BatchNormalization(name = f"{name_prefix}_bn2")(x)
    x = layers.ReLU(name = f"{name_prefix}_relu2")(x)
    return x


def decoder_block(inputs, skip, num_filters, name_prefix):
    x = layers.Conv2DTranspose(num_filters, (2,2), strides = (2,2), padding = "same", name = f"{name_prefix}_up")(inputs)
    x = layers.Concatenate(name = f"{name_prefix}_concat")([x , skip])
    x = conv_block(x, num_filters, name_prefix)
    return x

def building_u_net(input_shape = (Config.IMG_height, Config.IMG_width, Config.IMG_channel), num_classes = 1, base_filters = Config.base_filters):
    inputs = layers.Input(shape = input_shape, name = f"input_image")

    enc1 = conv_block(inputs, base_filters, name_prefix="ec1") #first encoder block
    p1 = layers.MaxPooling2D((2,2), name = "mp1")(enc1)

    enc2 = conv_block(p1, base_filters*2, name_prefix = "ec2")
    p2 = layers.MaxPooling2D((2,2), name = "mp2")(enc2)

    enc3 = conv_block(p2, base_filters*4, name_prefix="ec3")
    p3 = layers.MaxPooling2D((2,2), name = "mp3")(enc3)

    bn = conv_block(p3, base_filters*8, name_prefix="bottleneck")

    dec3 = decoder_block(bn, enc3, base_filters*4, name_prefix="dc3")
    dec2 = decoder_block(dec3, enc2, base_filters*2, name_prefix="dc2")
    dec1 = decoder_block(dec2, enc1, base_filters, name_prefix="dc1")

    output = layers.Conv2D(num_classes, (1,1), activation = "sigmoid", name = "output")(dec1)

    model = models.Model(inputs = [inputs], outputs = [output],name = "U-net")
    return model 
