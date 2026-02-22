import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small  # BETTER than V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from tensorflow.keras.regularizers import l2

DATASET_PATH = r"C:\Users\Admin\OneDrive\Desktop\Shell Project\Shell Project\dataset\train"
IMG_SIZE = 128
MAX_FRAMES = 16  # REDUCED for speed
EPOCHS = 100
BATCH_SIZE = 4

print("🚀 Starting High-Accuracy Violence Detection Training...")

# 🔥 ADVANCED DATA AUGMENTATION
def augment_frames(frames):
    """Apply augmentation to video frames"""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.2),
        A.ChannelShuffle(p=0.1)
    ])
    
    augmented_frames = []
    for frame in frames:
        # Convert to uint8 for augmentation
        frame_uint8 = (frame * 255).astype(np.uint8)
        augmented = transform(image=frame_uint8)['image']
        augmented_frames.append(augmented.astype(np.float32) / 255.0)
    
    return np.array(augmented_frames)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // MAX_FRAMES, 1)
    
    idx = 0
    count = 0
    while cap.isOpened() and count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx % step == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)
            count += 1
        idx += 1
    
    cap.release()
    
    # Pad with last frame instead of zeros (BETTER)
    while len(frames) < MAX_FRAMES:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    
    return np.array(frames)

# Load data
print("📂 Loading dataset...")
X, y = [], []
classes = ["non_violent", "violent"]

for label, cls in enumerate(classes):
    folder = os.path.join(DATASET_PATH, cls)
    if os.path.exists(folder):
        videos = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"Found {len(videos)} {cls} videos")
        
        for video in videos:
            try:
                frames = extract_frames(os.path.join(folder, video))
                # Apply augmentation to 70% of data
                if np.random.random() < 0.7:
                    frames = augment_frames(frames)
                X.append(frames)
                y.append(label)
            except Exception as e:
                print(f"Error processing {video}: {e}")

X = np.array(X)
y = np.array(y)
print(f"Total samples: {len(X)}, Shape: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Class weights (heavier penalty for violent misclassification)
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {0: 1.0, 1: 2.5}  # MORE weight to violent class
print(f"Class weights: {class_weights}")

# 🔥 BEST MODEL ARCHITECTURE
base_model = MobileNetV3Small(
    weights="imagenet", 
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Unfreeze more layers for fine-tuning
for layer in base_model.layers[:-15]:
    layer.trainable = False
for layer in base_model.layers[-15:]:
    layer.trainable = True

model = Sequential([
    TimeDistributed(base_model, input_shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3)),
    TimeDistributed(GlobalAveragePooling2D()),
    BatchNormalization(),
    
    # LSTM instead of GRU (BETTER for video)
    tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3),
    tf.keras.layers.LSTM(64, dropout=0.3),
    
    Dense(256, activation="swish", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(128, activation="swish", kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    
    Dense(1, activation="sigmoid")
])

# Compile with advanced optimizer
optimizer = Adam(learning_rate=0.0003, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# 🔥 BEST CALLBACKS
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
    ModelCheckpoint("best_violence_model.h5", monitor='val_accuracy', save_best_only=True)
]

print("🎯 Training BEST violence detection model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test)
print(f"\n🎉 FINAL RESULTS:")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"F1 Score: {(2*test_precision*test_recall)/(test_precision+test_recall):.4f}")

# Save final model
model.save("violence_model_perfect.h5")
print("✅ BEST MODEL SAVED: violence_model_perfect.h5")

# Confusion Matrix
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))
