#!/usr/bin/env python
# coding: utf-8

# # 🌿 Plant Disease Detection using CNN
# 
# Trains a CNN on the **PlantVillage** dataset for 3 plants:
# - 🥔 Potato — 3 classes
# - 🍅 Tomato — 10 classes
# - 🫑 Bell Pepper — 2 classes
# 
# **Total: 15 classes**
# 
# Dataset path: `../PlantVillage/`

# ## 1. Import Libraries

# In[ ]:


import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves plots instead of showing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

# Create output directory for plots
os.makedirs('training_outputs', exist_ok=True)
print("Plots will be saved to training_outputs/")

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ## 2. Configuration

# In[ ]:


DATASET_PATH = os.path.join(os.path.dirname(os.getcwd()), 'PlantVillage')
IMAGE_SIZE   = 256
BATCH_SIZE   = 32
EPOCHS       = 50
CHANNELS     = 3

# Plants to include — match actual folder name prefixes
SELECTED_PLANTS = ['Pepper__bell', 'Potato', 'Tomato']

print(f"Dataset path : {DATASET_PATH}")
print(f"Exists       : {os.path.exists(DATASET_PATH)}")


# ## 3. Explore Dataset

# In[ ]:


# Robust parser — handles _, __, ___ separators
PLANT_KEYS = [
    ('Pepper__bell', 'Bell Pepper'),
    ('Potato',       'Potato'),
    ('Tomato',       'Tomato'),
]

def parse_folder_name(folder):
    for plant_key, plant_disp in PLANT_KEYS:
        if folder.startswith(plant_key):
            rest    = folder[len(plant_key):].lstrip('_')
            disease = rest.replace('_', ' ').replace('  ', ' ').strip().title()
            return plant_key, plant_disp, disease or 'Unknown'
    return folder, folder.replace('_', ' ').title(), 'Unknown'

# Show only selected-plant folders
all_folders = sorted(os.listdir(DATASET_PATH))
selected_folders = [
    f for f in all_folders
    if os.path.isdir(os.path.join(DATASET_PATH, f))
    and any(f.startswith(p) for p in SELECTED_PLANTS)
]

print(f"Selected classes ({len(selected_folders)}):")
total_imgs = 0
prev_plant = None
for folder in selected_folders:
    plant_key, plant_disp, disease_disp = parse_folder_name(folder)
    if plant_key != prev_plant:
        print(f"\n  [{plant_disp}]")
        prev_plant = plant_key
    imgs = [f for f in os.listdir(os.path.join(DATASET_PATH, folder))
            if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"    {disease_disp:50s}: {len(imgs)} images")
    total_imgs += len(imgs)
print(f"\n  Total: {total_imgs} images")


# ## 4. Load Dataset (80 / 10 / 10 Split)

# In[ ]:


def get_dataset_partitions(ds, train_split=0.8, val_split=0.1, shuffle_size=10000):
    ds_size    = len(ds)
    ds         = ds.shuffle(shuffle_size, seed=42)
    train_size = int(train_split * ds_size)
    val_size   = int(val_split   * ds_size)
    return ds.take(train_size), ds.skip(train_size).take(val_size), ds.skip(train_size + val_size)

full_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    seed=42,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int'
)

CLASS_NAMES  = full_dataset.class_names
NUM_CLASSES  = len(CLASS_NAMES)

print(f"{NUM_CLASSES} classes detected:")
for i, c in enumerate(CLASS_NAMES):
    _, plant_disp, disease_disp = parse_folder_name(c)
    print(f"  {i:2d}: {plant_disp:12s} — {disease_disp}")

train_ds, val_ds, test_ds = get_dataset_partitions(full_dataset)
print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} batches")


# ## 5. Visualize Sample Images

# In[ ]:


plt.figure(figsize=(16, 12))
for images, labels in train_ds.take(1):
    for i in range(min(12, len(images))):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        _, plant_disp, disease_disp = parse_folder_name(CLASS_NAMES[labels[i]])
        plt.title(f"{plant_disp}\n{disease_disp}", fontsize=8)
        plt.axis('off')
plt.suptitle('Sample Training Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_outputs/1_sample_images.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: training_outputs/1_sample_images.png")


# ## 6. Class Distribution & Weights

# In[ ]:


all_labels = []
for _, labels in train_ds:
    all_labels.extend(labels.numpy())
all_labels = np.array(all_labels)

counts       = [int(np.sum(all_labels == i)) for i in range(NUM_CLASSES)]
short_labels = []
for c in CLASS_NAMES:
    _, pd, dd = parse_folder_name(c)
    short_labels.append(f"{pd[:3]}. {dd[:14]}")

plt.figure(figsize=(16, 5))
colors = plt.cm.Set3(np.linspace(0, 1, NUM_CLASSES))
plt.bar(range(NUM_CLASSES), counts, color=colors)
plt.xticks(range(NUM_CLASSES), short_labels, rotation=45, ha='right', fontsize=8)
plt.title('Class Distribution (Training Set)', fontsize=13)
plt.ylabel('Samples')
plt.tight_layout()
plt.savefig('training_outputs/2_class_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: training_outputs/2_class_distribution.png")

cw_array     = class_weight.compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = {i: w for i, w in enumerate(cw_array)}
print("Class weights computed.")


# ## 7. Data Augmentation & Preprocessing

# In[ ]:


resize_and_rescale = keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0 / 255),
], name='resize_rescale')

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name='augmentation')

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)
print("Preprocessing pipeline ready!")


# ## 8. Build CNN Model

# In[ ]:


model = keras.Sequential([
    resize_and_rescale,
    layers.Conv2D(32,  (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,  (3,3), activation='relu'),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,  (3,3), activation='relu'),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,  (3,3), activation='relu'),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'), layers.Dropout(0.4),
    layers.Dense(64,  activation='relu'), layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax'),
], name='plant_disease_cnn')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
model.build((None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
model.summary()


# ## 9. Train the Model

# In[ ]:


os.makedirs('saved_models', exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint('saved_models/best_plant_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, class_weight=class_weights
)


# ## 10. Training History

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],     label='Train',      color='#3498db')
axes[0].plot(history.history['val_accuracy'], label='Validation', color='#e74c3c')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['loss'],     label='Train',      color='#3498db')
axes[1].plot(history.history['val_loss'], label='Validation', color='#e74c3c')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle('Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_outputs/3_training_history.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: training_outputs/3_training_history.png")
print(f"Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")


# ## 11. Evaluate on Test Set

# In[ ]:


scores = model.evaluate(test_ds)
print(f"\nTest Loss     : {scores[0]:.4f}")
print(f"Test Accuracy : {scores[1]*100:.2f}%")


# ## 12. Confusion Matrix & Report

# In[ ]:


y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())
y_true = np.array(y_true); y_pred = np.array(y_pred)

short_names = [f"{pd[:3]}. {dd[:15]}" for _, pd, dd in [parse_folder_name(c) for c in CLASS_NAMES]]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=short_names, yticklabels=short_names)
plt.title('Confusion Matrix — Test Set', fontsize=13)
plt.ylabel('True'); plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('training_outputs/4_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: training_outputs/4_confusion_matrix.png")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=short_names))


# ## 13. Sample Predictions

# In[ ]:


plt.figure(figsize=(16, 12))
idx = 0
for images, labels in test_ds.take(3):
    preds = model.predict(images, verbose=0)
    for i in range(len(images)):
        if idx >= 12: break
        ax = plt.subplot(3, 4, idx + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        pred_idx = np.argmax(preds[i]); true_idx = labels[i].numpy(); conf = preds[i][pred_idx]
        _, tp, td = parse_folder_name(CLASS_NAMES[true_idx])
        _, pp, pd = parse_folder_name(CLASS_NAMES[pred_idx])
        plt.title(f"True:{tp}\n{td}\nPred:{pp}\n{pd}({conf:.0%})",
                  color='green' if pred_idx==true_idx else 'red', fontsize=7)
        plt.axis('off'); idx += 1
    if idx >= 12: break
plt.suptitle('Predictions (Green=Correct, Red=Wrong)', fontsize=13)
plt.tight_layout()
plt.savefig('training_outputs/5_sample_predictions.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: training_outputs/5_sample_predictions.png")


# ## 14. Save Model

# In[ ]:


os.makedirs('saved_models', exist_ok=True)
model.save('saved_models/plant_model.h5')
print("Saved -> saved_models/plant_model.h5")

with open('saved_models/class_names.txt', 'w') as f:
    for cn in CLASS_NAMES: f.write(cn + '\n')
print("Saved -> saved_models/class_names.txt")

backend_dir = os.path.join(os.path.dirname(os.getcwd()), 'backend', 'models')
os.makedirs(backend_dir, exist_ok=True)
shutil.copy('saved_models/plant_model.h5',  os.path.join(backend_dir, 'plant_model.h5'))
shutil.copy('saved_models/class_names.txt', os.path.join(backend_dir, 'class_names.txt'))
print(f"Copied to {backend_dir}")

print(f"\n{NUM_CLASSES} classes:")
for i, c in enumerate(CLASS_NAMES):
    _, pd, dd = parse_folder_name(c)
    print(f"  {i:2d}: {pd:12s} — {dd}")


# ## 15. Quick Inference Test

# In[ ]:


from PIL import Image as PILImage

def predict_image(img_path, model, class_names, image_size=256):
    img     = PILImage.open(img_path).convert('RGB')
    img_arr = np.expand_dims(np.array(img.resize((image_size, image_size))), axis=0)
    preds   = model.predict(img_arr, verbose=0)
    idx     = np.argmax(preds[0])
    conf    = preds[0][idx]
    _, plant_disp, disease_disp = parse_folder_name(class_names[idx])
    print(f"Plant     : {plant_disp}")
    print(f"Disease   : {disease_disp}")
    print(f"Confidence: {conf:.2%}")
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(f"{plant_disp}\n{disease_disp} ({conf:.1%})",
              color='green' if 'healthy' in class_names[idx].lower() else 'red')
    plt.axis('off')
    plt.savefig('training_outputs/6_inference_test.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Saved: training_outputs/6_inference_test.png")

# Auto-test: pick first image from dataset
sample_class = os.listdir(DATASET_PATH)[0]
sample_img   = os.path.join(DATASET_PATH, sample_class,
                            os.listdir(os.path.join(DATASET_PATH, sample_class))[0])
print(f"Testing: {sample_img}\n")
predict_image(sample_img, model, CLASS_NAMES)

# Your own image:
# predict_image('/path/to/leaf.jpg', model, CLASS_NAMES)

