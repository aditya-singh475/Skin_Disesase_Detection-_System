from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Load validation data
val_gen = ImageDataGenerator(rescale=1./255)
val_data = val_gen.flow_from_directory(
    "data/val_resized",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ✅ Load trained model
model = load_model("model/models/skin_model.h5")

# ✅ Predict
preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

# ✅ Load class names
with open("model/models/class_indices.json") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# ✅ Print report
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("📊 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# ✅ Optional: Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/models/confusion_matrix.png")
print("✅ Confusion matrix saved as model/models/confusion_matrix.png")
