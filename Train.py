import cv2
import numpy as np
from PIL import Image
import sqlite3

data = []
label = []   

for i in range (1,6):
    for j in range(1, 21):
        filename = './dataSet/User.' + str(i) + '.' + str(j) + '.jpg'
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(src=img, dsize=(100, 100))
        img = np.array(img)
        data.append(img)
        label.append(i - 1)
data1 = np.array(data)
label = np.array(label)
data1 = data1.reshape((100, 100, 100, 1))
X_train = data1/255
#-----------------------------------------------------------------------------
# Chuyển đổi các labels thành dạng nhị phân
from sklearn.preprocessing import LabelBinarizer

Y_train = LabelBinarizer().fit_transform(label)
#-----------------------------------------------------------------------------
from tensorflow import keras
from tensorflow.keras.layers import (Activation, BatchNormalization, Dropout, AveragePooling2D, Conv2D, Dense, Flatten, Input, MaxPooling2D, concatenate)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

shape = (100,100, 1)

Model = Sequential()

Model.add(Conv2D(32, (3, 3), padding="same", input_shape = shape))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size = (2, 2)))#

Model.add(Conv2D(32, (3, 3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size = (2, 2)))

Model.add(Conv2D(64, (3, 3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size = (2, 2)))

Model.add(Flatten())

Model.add(Dense(512))
Model.add(Activation("relu"))
Model.add(Dense(5))
Model.add(Activation('softmax'))

Model.summary()
Model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#-----------------------------------------------------------------------------
print("Start training...")

Model.fit(X_train, Y_train, batch_size = 5, epochs = 10)
Model.save("recognizer/trainer.h5")
