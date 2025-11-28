import tensorflow as tf
from config import Config


def flatten_tensor(y_true, y_pred): 
    y_true = tf.cast(y_true, "float32") #converting the data type to float32
    y_pred = tf.cast(y_pred, "float32") #converting the data type to float32

    if tf.rank(y_true) == 3: #if the dimension of the true is about 3
        y_true = tf.expand_dims(y_true, axis = -1)
    if tf.rank(y_pred) == 3: 
        y_pred = tf.expand_dims(y_pred, axis = -1)
    
    #flattening the tensor now 
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))

    return y_true, y_pred


@tf.keras.utils.register_keras_serializable(name = "lesion_metrics")
def dice_coeff(y_true, y_pred):
    """
    Dice coefficienct = (2 * A intersection B)/ |A| + |B|
    """
    y_true, y_pred = flatten_tensor(y_true, y_pred) #flattening the tensor 
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_true = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_coefficient = (2 * intersection + Config._EPS)/(sum_true + Config._EPS) #dice coefficient
    return dice_coefficient

@tf.keras.utils.register_keras_serializable(name = "lesion_metrics")
def iou_metric(y_true, y_pred):
    """
    IOU = (A intersection B)/(A Union B)
    """
    y_true, y_pred = flatten_tensor(y_true, y_pred) #flattenng 
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    iou = (intersection)/(union + Config._EPS)

    return iou
