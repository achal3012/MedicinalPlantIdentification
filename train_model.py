import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

DATASET_PATH = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 4

mobile_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
mobile_train_data = mobile_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

mobile_validation_data = mobile_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("MobileNetV2 Classes:", mobile_train_data.class_indices)
print("MobileNetV2 Training images:", mobile_train_data.samples)
print("MobileNetV2 Validation images:", mobile_validation_data.samples)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)
model2 = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(9, activation="softmax")
])
model2.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
checkpoint = ModelCheckpoint(
    "model/medicinal_plant_mobilenetv2.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
history2 = model2.fit(
    mobile_train_data,
    validation_data=mobile_validation_data,
    epochs=5,
    callbacks=[checkpoint]
)

print("MobileNetV2 model saved successfully!")
best_mobilenet_model = tf.keras.models.load_model(
    "model/medicinal_plant_mobilenetv2.keras"
)

mobile_loss, mobile_accuracy = best_mobilenet_model.evaluate(
    mobile_validation_data
)

print("Best MobileNetV2 Validation Accuracy:", mobile_accuracy)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("Classes:", train_data.class_indices)
print("Training images:", train_data.samples)
print("Validation images:", validation_data.samples)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_checkpoint = ModelCheckpoint(
    "model/medicinal_plant_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

print("CNN model created successfully!")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10,
    callbacks=[cnn_checkpoint]
)

print("CNN training completed!")

print("CNN model saved successfully!")

best_cnn_model = tf.keras.models.load_model(
    "model/medicinal_plant_cnn.keras"
)

loss, accuracy = best_cnn_model.evaluate(validation_data)

print("Best CNN Validation Accuracy:", accuracy)