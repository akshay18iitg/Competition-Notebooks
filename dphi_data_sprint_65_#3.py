#IMPORTING PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow
import os
import sklearn
import sklearn.model_selection
from tensorflow.keras.applications import Xception,ResNet50,InceptionResNetV2,EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7
from tensorflow.keras.layers import Input,Dense,GlobalAveragePooling2D,BatchNormalization,Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


data1 = pd.read_csv('./Training_set.csv')
data1['label'].replace(['cruise_ship', 'gondola', 'buoy', 'sailboat', 'ferry_boat',
       'inflatable_boat', 'kayak', 'paper_boat', 'freight_boat'],[0,1,2,3,4,5,6,7,8],inplace = True)
sns.countplot(data1['label'])
lst1 = data1.label.unique()

#Controlled Data Augmentation
X = []
y = []
data2 = data1[data1['label'] == lst1[0]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X.append(img1)
    y.append(data2['label'][i])
    img2 = cv2.flip(img1,1)
    X.append(img2)
    y.append(data2['label'][i])
data2 = data1[data1['label'] == lst1[1]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X.append(img1)
    y.append(data2['label'][i])
    img2 = cv2.flip(img1,1)
    X.append(img2)
    y.append(data2['label'][i])
data2 = data1[data1['label'] == lst1[2]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X.append(img1)
    y.append(data2['label'][i])
    img2=img1+50
    X.append(img2)
    y.append(data2['label'][i])
    img3 = cv2.flip(img1,1)
    X.append(img3)
    y.append(data2['label'][i])
    img4 = cv2.flip(img2,1)
    X.append(img4)
    y.append(data2['label'][i])
    img5=img1-50
    X.append(img5)
    y.append(data2['label'][i])
data2 = data1[data1['label'] == lst1[3]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X.append(img1)
    y.append(data2['label'][i])
data2 = data1[data1['label'] == lst1[4]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X.append(img1)
    y.append(data2['label'][i])
    img2=img1+50
    X.append(img2)
    y.append(data2['label'][i])
    img3 = cv2.flip(img1,1)
    X.append(img3)
    y.append(data2['label'][i])
    img4 = cv2.flip(img2,1)
    X.append(img4)
    y.append(data2['label'][i])
    img5=img1-50
    X.append(img5)
    y.append(data2['label'][i])
data2 = data1[data1['label'] == lst1[5]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X.append(img1)
    y.append(data2['label'][i])
    img2=img1+50
    X.append(img2)
    y.append(data2['label'][i])
    img3 = cv2.flip(img1,1)
    X.append(img3)
    y.append(data2['label'][i])
    img4 = cv2.flip(img2,1)
    X.append(img4)
    y.append(data2['label'][i])
    img5=img1-50
    X.append(img5)
    y.append(data2['label'][i])
    img6=cv2.flip(img5,1)
    X.append(img6)
    y.append(data2['label'][i])
    img7=img1+30
    X.append(img7)
    y.append(data2['label'][i])
    img8=cv2.flip(img7,1)
    X.append(img8)
    y.append(data2['label'][i])
data2 = data1[data1['label'] == lst1[6]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    img3 = cv2.flip(img1,1)
    X.append(img3)
    y.append(data2['label'][i])
data2 = data1[data1['label'] == lst1[7]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X.append(img1)
    y.append(data2['label'][i])
    img2=img1+50
    X.append(img2)
    y.append(data2['label'][i])
    img3 = cv2.flip(img1,1)
    X.append(img3)
    y.append(data2['label'][i])
    img4 = cv2.flip(img2,1)
    X.append(img4)
    y.append(data2['label'][i])
    img5=img1-50
    X.append(img5)
    y.append(data2['label'][i])
    img6=cv2.flip(img5,1)
    X.append(img6)
    y.append(data2['label'][i])
    img7=img1+30
    X.append(img7)
    y.append(data2['label'][i])
    img8=cv2.flip(img7,1)
    X.append(img8)
    y.append(data2['label'][i])
data2 = data1[data1['label'] == lst1[8]].reset_index()
for i in range(len(data2)):
    img1 = cv2.imread('./train/' + data2['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X.append(img1)
    y.append(data2['label'][i])
    img2=img1+50
    X.append(img2)
    y.append(data2['label'][i])
    img3 = cv2.flip(img1,1)
    X.append(img3)
    y.append(data2['label'][i])
    img4 = cv2.flip(img2,1)
    X.append(img4)
    y.append(data2['label'][i])
    img5=img1-50
    X.append(img5)
    y.append(data2['label'][i])
    img6=cv2.flip(img5,1)
    X.append(img6)
    y.append(data2['label'][i])
    img7=img1+30
    X.append(img7)
    y.append(data2['label'][i])
    img8=cv2.flip(img7,1)
    X.append(img8)
    y.append(data2['label'][i])
y = np.array(y)
sns.countplot(y)
y = tensorflow.keras.utils.to_categorical(y)
X = np.array(X)
X_train,X_val,y_train,y_val = sklearn.model_selection.train_test_split(X,y,test_size = 0.1,random_state = 30)
train_gen = tensorflow.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2,
    height_shift_range=0.2, brightness_range=(0.3,0.7),zoom_range=0.5,
    horizontal_flip=True)
val_gen = tensorflow.keras.preprocessing.image.ImageDataGenerator()
scheduler = tensorflow.keras.optimizers.schedules.CosineDecay(0.0001,100)
inputs = Input(shape = (224,224,3))
base_model = EfficientNetB7(include_top = False)(inputs)
x = GlobalAveragePooling2D()(base_model)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256,activation = 'leaky_relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(9,activation = 'softmax')(x)
model = Model(inputs,outputs)
model.compile(loss = 'categorical_crossentropy',optimizer = Adam(lr = 0.001),metrics = ['accuracy'])
X_train = X[:700]
X_val = X[700:]
y_train = tensorflow.keras.utils.to_categorical(data1['label'][:700],9)
y_val = tensorflow.keras.utils.to_categorical(data1['label'][700:],9)
my_callbacks = [tensorflow.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_accuracy:.2f}.h5',save_best_only = True),tensorflow.keras.callbacks.LearningRateScheduler(scheduler)]
model.fit(X_train,y_train,epochs = 100,validation_data = (X_val,y_val),callbacks = my_callbacks,batch_size = 16)
test1 = pd.read_csv('./Testing_set.csv')
X_test = []
for i in range(len(test1)):
    img1 = cv2.imread('./test/' + test1['filename'][i])
    img1 = cv2.resize(img1,(224,224))
    img1.shape
    X_test.append(img1)
X_test = np.array(X_test)
model = tensorflow.keras.models.load_model('./model.25-0.97.h5')
y_pred = np.argmax(model.predict(X_test),axis = 1)
test1['label'] = y_pred
test1['label'].replace([0,1,2,3,4,5,6,7,8],['cruise_ship', 'gondola', 'buoy', 'sailboat', 'ferry_boat',
       'inflatable_boat', 'kayak', 'paper_boat', 'freight_boat'],inplace = True)
test1.to_csv('sub1.csv')
