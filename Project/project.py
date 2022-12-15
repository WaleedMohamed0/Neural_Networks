import glob
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
import matplotlib.pyplot as plt

IMG_SIZE = 120
LR = 1e-3
TRAIN_DIR = 'Train/'
TEST_DIR = 'Test/'

def convertToJPG(path):
    for i in os.listdir(path):
        os.rename(os.path.join(path,i),os.path.join(path,os.path.splitext(i)[0] + '.jpg'))

def create_label(image_name):
    if image_name.__contains__('Basketball'):
        return np.array([1,0,0,0,0,0])
    elif image_name.__contains__('Football'):
        return np.array([0,1,0,0,0,0])
    elif image_name.__contains__('Rowing'):
        return np.array([0,0,1,0,0,0])
    elif image_name.__contains__('Swimming'):
        return np.array([0,0,0,1,0,0])
    elif image_name.__contains__('Tennis'):
        return np.array([0,0,0,0,1,0])
    elif image_name.__contains__('Yoga'):
        return np.array([0,0,0,0,0,1])
    else:
        print("Error in label creation")

def create_data(dir):
    data = []
    for img in tqdm(os.listdir(dir)):
        path = os.path.join(dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img), create_label(path)])
    shuffle(data)
    np.save('train_data.npy', data)
    return data

def augment_data(dir):
    data = []
    for img in tqdm(os.listdir(dir)):
        path = os.path.join(dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img), create_label(path)])
        data.append([np.array(cv2.flip(img, 1)), create_label(path)])
        data.append([np.array(cv2.flip(img, -1)), create_label(path)])
        data.append([np.array(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)), create_label(path)])
        data.append([np.array(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)), create_label(path)])
        data.append([np.array(cv2.rotate(img, cv2.ROTATE_180)), create_label(path)])
    shuffle(data)
    np.save('drive/My Drive/Colab Notebooks/train_data.npy', data)
    return data
    

if (os.path.exists('train_data.npy')):
    train_data = np.load('train_data.npy',allow_pickle=True)
else:
    train_data = augment_data(TRAIN_DIR)


X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = [i[1] for i in train_data]
y_train = np.array(y_train)
X_train = X_train / 255.0

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

if(os.path.exists('sports.h5')):
    model = tf.keras.models.load_model('sports.h5')
else:
    early_stopping = EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])
    model.save('sports.h5')


with open('submission_file.csv', 'w') as f:
    f.write('image_name,label\n')
    for i in os.listdir('Test'):
        img = cv2.imread('Test/' + i, cv2.IMREAD_COLOR)
        test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 3)
        test_img = test_img / 255.0

        prediction = model.predict(np.array([test_img]))
        if(np.argmax(prediction) == 0):
            f.write(i + ',' + '0')
        elif(np.argmax(prediction) == 1):
            f.write(i + ',' + '1')
        elif(np.argmax(prediction) == 2):
            f.write(i + ',' + '2')
        elif(np.argmax(prediction) == 3):
            f.write(i + ',' + '3')
        elif(np.argmax(prediction) == 4):
            f.write(i + ',' + '4')
        elif(np.argmax(prediction) == 5):
            f.write(i + ',' + '5')
        f.write('\n')
