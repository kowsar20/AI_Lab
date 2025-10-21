import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

# Load your dataset

shirt_path = "ResizedShirt"
tshirt_path = "ResizedTShirt"

img_height, img_width = 224, 224  # Standard VGG input size

def load_images(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = load_img(path, target_size=(img_height, img_width))
            img = img_to_array(img)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return images, labels

shirt_imgs, shirt_lbls = load_images(shirt_path, 0)
tshirt_imgs, tshirt_lbls = load_images(tshirt_path, 1)

X = np.array(shirt_imgs + tshirt_imgs, dtype="float32") / 255.0
Y = np.array(shirt_lbls + tshirt_lbls)

# Shuffle and split data
X, Y = shuffle(X, Y, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, "Test:", X_test.shape)


inputs = Input(shape=(224, 224, 3), name="input_layer")

# Block 1
x = Conv2D(64, (3,3), activation="relu", padding="same")(inputs)
x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D((2,2))(x)

# Block 2
x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D((2,2))(x)

# Block 3
x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D((2,2))(x)

# Block 4
x = Conv2D(512, (3,3), activation="relu", padding="same")(x)
x = Conv2D(512, (3,3), activation="relu", padding="same")(x)
x = MaxPooling2D((2,2))(x)

# Flatten + Fully Connected Layers
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs, outputs, name="VGG16_Custom")


# Compile and Train

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=1
)


# Evaluate Model

loss, acc = model.evaluate(X_test, Y_test)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")


# Plot Accuracy and Loss

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig("vgg16_accuracy.png")
plt.show()


# Visualize Predictions

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
indices = random.sample(range(len(X_test)), 12)

plt.figure(figsize=(12,8))
for i, idx in enumerate(indices):
    plt.subplot(3,4,i+1)
    plt.imshow(X_test[idx])
    plt.axis("off")
    pred = "TShirt" if y_pred[idx]==1 else "Shirt"
    actual = "TShirt" if Y_test[idx]==1 else "Shirt"
    plt.title(f"Pred: {pred}\nActual: {actual}")
plt.tight_layout()
plt.savefig("vgg16_predictions.png")
plt.show()
