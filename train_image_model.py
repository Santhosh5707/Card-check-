"""
Train a card-image screening classifier using transfer learning.
Expected folders:
  data/card_images/train/genuine/
  data/card_images/train/counterfeit/
  data/card_images/validation/genuine/
  data/card_images/validation/counterfeit/

Use only images you have permission to use. Synthetic card images are not evidence of real-world counterfeit detection.
Run in Colab or a Python environment with TensorFlow installed:
  pip install tensorflow
  python train_image_model.py
"""
from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "card_images"
TRAIN = DATA / "train"
VAL = DATA / "validation"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

for folder in [TRAIN / "genuine", TRAIN / "counterfeit", VAL / "genuine", VAL / "counterfeit"]:
    if not folder.exists() or not any(folder.iterdir()):
        raise SystemExit(
            f"Missing labeled image folder or it is empty: {folder}\n"
            "Do not use randomly labeled or synthetic images as if they were real counterfeit examples."
        )

IMG_SIZE = (224, 224)
BATCH = 16
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN, image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary",
    class_names=["genuine", "counterfeit"], shuffle=True, seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL, image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary",
    class_names=["genuine", "counterfeit"], shuffle=False
)
# Labels: genuine=0, counterfeit=1
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base.trainable = False
inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall"),
             tf.keras.metrics.AUC(name="roc_auc")]
)
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
model.save(OUT / "card_image_model.keras")
(OUT / "card_image_metrics.json").write_text(json.dumps({
    "class_names": ["genuine", "counterfeit"],
    "image_size": [224, 224],
    "metrics_by_epoch": metrics,
    "warning": "Validation metrics on supplied dataset do not prove real-world authenticity performance."
}, indent=2), encoding="utf-8")
print("Saved models/card_image_model.keras")
print("Review validation precision/recall and test on an independent, representative dataset before use.")
