# src/train_model.py

import os, json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ===== Config =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DIR = "data/train_resized"
VAL_DIR   = "data/val_resized"
SAVE_DIR  = "model/models"
MODEL_PATH = os.path.join(SAVE_DIR, "skin_model.h5")
CLASS_INDICES_PATH = os.path.join(SAVE_DIR, "class_indices.json")
HISTORY_PATH = os.path.join(SAVE_DIR, "training_history.json")

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== Data Generators =====
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Save class mapping once
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_data.class_indices, f)

# ===== CNN Model (light yet solid) =====
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===== Callbacks =====
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
]

# ===== Train =====
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save training history to JSON
hist = {k: [float(x) for x in v] for k, v in history.history.items()}
with open(HISTORY_PATH, "w") as f:
    json.dump(hist, f)

print(f"✅ Training complete.\n- Model: {MODEL_PATH}\n- Class indices: {CLASS_INDICES_PATH}\n- History: {HISTORY_PATH}")


