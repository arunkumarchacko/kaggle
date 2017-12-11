import numpy as np
import pprint as pp
import os
import glob as glob
import pandas as pd
from PIL import Image

import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Flatten
from keras.utils import to_categorical

dataDir = '/home/arun/ml/data/kaggledata/dogbreed/'
trainDir = os.path.join(dataDir, 'train')
testDir = os.path.join(dataDir, 'test')

labelPath = os.path.join(dataDir, 'labels.csv')

# print(trainDir, testDir, )

trainFiles = glob.glob(os.path.join(trainDir, '*.jpg'))[:100000]
testFiles = glob.glob(os.path.join(testDir, '*.jpg'))
labels = pd.read_csv(labelPath)

print('Train:', len(trainFiles), 'Test:', len(testFiles), 'Labels:', len(labels))


# print(labels.head())
# print(labels.describe())

imgs = np.array([np.array(Image.open(file).resize((224,224))) for file in trainFiles])

ids = [file.split('/')[-1].split('.')[0] for file in trainFiles]
id_val = dict(zip(labels.id,labels.breed))
label_ids = [id_val[id] for id in ids]
label_index = {label:i for i,label in enumerate(np.unique(label_ids))}
labels_id = [label_index[label] for label in label_ids]

# pp.pprint(label_ids)
# pp.pprint(label_index)
# pp.pprint(labels_id)

labels_onehot_encoded = to_categorical(labels_id,num_classes=120)
base_model = VGG16(include_top=False,weights=None,input_shape=(224,224,3))

model = Sequential(base_model.layers[:7])

model.add(Flatten())
model.add(Dense(512,activation='relu'))
#model.add(Dense(256,activation='relu'))
model.add(Dense(120,activation='softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(imgs,labels_onehot_encoded,batch_size=8,validation_split=0.2,)
