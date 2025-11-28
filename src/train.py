import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from config import Config
from metrics import iou_metric, dice_coeff
from dataset import loading_dataset
from model import building_u_net
from metrics import iou_metric, dice_coeff
import argparse

parser = argparse.ArgumentParser(description="Train U-Net model for image segmentation")

# Use None as default so we can detect if the user passed the argument or not
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate for the optimizer')
parser.add_argument('--model_path', type=str, default=None, help='Path to save the trained model')
parser.add_argument('--data_path', type=str, default=None, help='Path to the training data')

args = parser.parse_args()

# ---- Override Config ONLY if the user provided a value ----
if args.epochs is not None:
    Config.num_epochs = args.epochs

if args.batch_size is not None:
    Config.batch_size = args.batch_size

if args.learning_rate is not None:
    Config.learning_rate = args.learning_rate

if args.model_path is not None:
    Config.model_dir = args.model_path

if args.data_path is not None:
    Config.Data_dir = args.data_path

# ----------------- From here on, just use Config -----------------

# Loading the dataset
train_ds, val_ds = loading_dataset()

# Initialize the model 
model = building_u_net(
    input_shape=(Config.IMG_height, Config.IMG_width, Config.IMG_channel),
    num_classes=1, base_filters=Config.base_filters
)

loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=Config.learning_rate)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        dice_coeff,
        iou_metric,
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# Defining the callbacks
checkpoints = os.path.join(Config.model_dir, "best_model_lesion")
callbacks = [
    ModelCheckpoint(
        filepath=checkpoints,
        monitor="val_dice_coef",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_dice_coef",
        patience=10,
        mode="max",
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    )
]

# Training the model 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=Config.num_epochs,
    callbacks=callbacks,
    verbose=1
)

# Saving the final model
final_path = os.path.join(Config.model_dir, "final_model.keras")
model.save(final_path)
print(f"Training complete. Model saved to: {final_path}")
