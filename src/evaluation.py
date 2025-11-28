import tensorflow as tf
from dataset import loading_dataset
from config import Config
import os
from metrics import iou_metric, dice_coeff
from matplotlib import pyplot as plt
import numpy as np 


"""loading the model that we jsut saved after training"""

_ , val_ds = loading_dataset() #acquiring the validation dataset
best_model_path = os.path.join(Config.model_dir, "best_model_lesion")

"""Loading the model """
model = tf.keras.load_model(best_model_path, custom_objects = {
    "iou_metric" : iou_metric,
    "dice_coeff" : dice_coeff
})


"Now the evluation begin"
results = model.evaluate(val_ds)

for name, score in zip(model.metric_names, results):
    print(f"name: {name}, score: {score: .4f}")

def visualize_predictions(dataset, model, num_samples=3):
    for images, masks in dataset.take(1):
        preds = model.predict(images)
        preds = (preds > 0.5).astype(np.float32)  # threshold at 0.5

        for i in range(num_samples):
            plt.figure(figsize=(10,3))
            plt.subplot(1,3,1)
            plt.imshow(images[i, :, :, 0], cmap="gray")
            plt.title("Input MRI")
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.imshow(masks[i, :, :, 0], cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.imshow(images[i, :, :, 0], cmap="gray")
            plt.imshow(preds[i, :, :, 0], cmap="Reds", alpha=0.4)
            plt.title("Prediction Overlay")
            plt.axis("off")

            plt.show()

visualize_predictions(val_ds, model, num_samples=5)

