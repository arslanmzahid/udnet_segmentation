import numpy as np 
import os

class Config: 
    Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the directory to the master 
    Data_dir = os.path.join(Base_dir, "data")
    Processed_dir = os.path.join(Data_dir, "preprocess") #directory for the processed data
    Output_dir = os.path.join(Base_dir, "outputs") #the output directory 



    #now adding model parameters for the model to work 
    num_epochs = 10 
    learning_rate = 1e-4
    batch_size = 20
    validation_split = 0.2
    base_filters = 64

    #now adding the augmentation side
    augmentation = {
        "flip_horizontal" : True, 
        "flip_vertical" : True, 
        "rotation_angle" : 10, 
        "zoom" : 0.1
    }

    #image parameters that are going to be fed to the model: 
    IMG_height = 128
    IMG_width = 128 
    IMG_channel = 1
    image_size = (IMG_height, IMG_width, IMG_channel)

    shuffle = True


    _EPS = 1e-4
    prefix = "train"

    #saving model directory 
    model_dir = "best_model_params"
    saving_dir = os.path.join(Output_dir, model_dir)
    
    # Data path for train.py argument compatibility
    Data_path = Data_dir

