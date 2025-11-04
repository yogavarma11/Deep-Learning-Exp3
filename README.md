# Experiment-3 CONVOLUTIONAL DEEP NEURAL NETWORK FOR IMAGE CLASSIFICATION

## AIM

To develop a Convolutional Deep Neural Network (CNN) for image classification and to verify its performance on custom images.

## PROBLEM STATEMENT AND DATASET

**Problem Statement**

The objective is to design and implement a Convolutional Neural Network that can classify input images into multiple categories. The CNN should automatically extract meaningful features from raw image data and use them to distinguish between different classes (e.g., animals, objects, digits). It must also demonstrate the ability to generalize to real-world images not present in the training dataset.

**Dataset**

For experimentation, we use the CIFAR-10 dataset, a widely used benchmark in computer vision. It consists of 60,000 color images of size 32×32 pixels, divided into 10 categories such as airplanes, cars, birds, cats, and dogs. The dataset is split into:

50,000 training images

10,000 testing images

Pixel intensities range from 0 to 255, where 0 represents black and 255 represents white. By normalizing these values and applying one-hot encoding to labels, the dataset becomes suitable for CNN training and evaluation.


# Neural Network Model

<img width="1120" height="631" alt="image" src="https://github.com/user-attachments/assets/9b4c2f9f-d4e5-41e4-a62a-e49253f99088" />


## DESIGN STEPS

STEP 1:

Preprocess the CIFAR-10 dataset by scaling pixel values to [0, 1] and applying one-hot encoding to labels.

STEP 2:

Build a CNN model with the following layers:

Input → Convolution → MaxPooling → Convolution → MaxPooling → Flatten → Dense layers → Output.

STEP 3:

Compile the model using categorical cross-entropy loss and the Adam optimizer.

STEP 4:

Train the CNN for multiple epochs (e.g., 10) with a batch size of 64.

STEP 5:

Evaluate the model on the test set by analyzing accuracy, loss curves, confusion matrix, and classification report. Test predictions on custom images.


## PROGRAM

**Name:** YOGAVARMA B

**Register Number:** 2305002029

```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

single_image= X_train[0]
single_image.shape

plt.imshow(single_image,cmap='gray')
y_train.shape

X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('7.png')

type(img)

img = image.load_img('7.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

```

## OUTPUT

**Training data**

<img width="1148" height="286" alt="image" src="https://github.com/user-attachments/assets/5485841c-c9d2-45ba-b74a-123aa5a0beab" />


**Training Loss, Validation Loss Vs Iteration Plot:**


<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/3fbb4ed5-4684-4d04-8649-d697240ca895" />


![WhatsApp Image 2025-09-11 at 14 40 18_0de99ca6](https://github.com/user-attachments/assets/578c829b-6079-4e57-aec6-1cf23a154c96)


**Confusion Matrix**

<img width="569" height="336" alt="image" src="https://github.com/user-attachments/assets/e9df23bd-89ee-4c50-9e84-dd9912d4b6fc" />


**Classification Report**

<img width="573" height="212" alt="image" src="https://github.com/user-attachments/assets/fcfbcebe-26f3-43cc-80aa-0e1f748c5efc" />


**New Sample Data Prediction**


<img width="416" height="413" alt="image" src="https://github.com/user-attachments/assets/f7fef4f5-c919-4e10-9936-288af55a581f" />



<img width="416" height="413" alt="image" src="https://github.com/user-attachments/assets/c069db17-ef96-4092-8f15-d19ed84ece0a" />


## RESULT

Thus, a Convolutional Deep Neural Network for digit classification was developed and successfully verified with both MNIST dataset and scanned handwritten images.
