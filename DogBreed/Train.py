import sys
sys.path.append('../')
import utils as utils


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
from keras import optimizers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import pickle as pickle

home = utils.GetHomeDir()

originalTrainDir = os.path.join(home, 'ml/data/kaggle/dogbreed/train/')
originalTestDir = os.path.join(home, 'ml/data/kaggle/dogbreed/test/')
# originalValidDir = os.path.join(home, 'ml/data/kaggle/dogbreed/valid/')
labelPath = os.path.join(home, 'ml/data/kaggle/dogbreed/labels.csv')

outPath = os.path.join(home, 'temp/dog_breed_workingdir')
modelPath = os.path.join(home, 'ml/data/kaggle/dogbreed/model2.h5')
featuresPath = os.path.join(home, 'ml/data/kaggle/dogbreed/features.pkl')


# utils.CreateDir(trainDir)
# utils.CreateDir(validDir)
# print('TrainPath:', trainDir, 'Validation:', validDir)


utils.SetupKaggleData(originalTrainDir, labelPath,  outPath, 'breed', originalTestDir)
# utils.SetupKaggleData(originalValidDir, labelPath,  validDir, 'breed')


trainDir = os.path.join(outPath, 'train')
validDir = os.path.join(outPath, 'valid')
# trainDir = os.path.join(outPath, 'trial')

nb_train_samples = len(utils.GetImageFilesInDir(trainDir))
nb_validation_samples = len(utils.GetImageFilesInDir(validDir))
epochs = 10
batch_size = 8
img_width, img_height = 224, 224

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

for l in base_model.layers[:-6]:
    l.trainable = False
    # print(l.name)

x = Flatten()(base_model.output)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(120, activation = 'sigmoid')(x)

model = Model(input = base_model.input, output = predictions)

# print(model.summary())
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(trainDir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

with open(featuresPath, 'wb') as handle:
    pp.pprint(train_generator.class_indices)
    pickle.dump(train_generator.class_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

validation_generator = test_datagen.flow_from_directory(validDir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, epochs=epochs, validation_data=validation_generator, nb_val_samples=nb_validation_samples)

print('Saving model to:', modelPath)
model.save(modelPath)


























# # dataDir = '/home/arun/ml/data/kaggledata/dogbreed/'
# # modelPath = '/home/arun/ml/data/kaggledata/model1.h5'
# # trainDir = os.path.join(dataDir, 'train')
# # testDir = os.path.join(dataDir, 'test')

# # labelPath = os.path.join(dataDir, 'labels.csv')

# # # print(trainDir, testDir, )

# # trainFiles = glob.glob(os.path.join(trainDir, '*.jpg'))[:100000]
# # testFiles = glob.glob(os.path.join(testDir, '*.jpg'))
# # labels = pd.read_csv(labelPath)

# # print('Train:', len(trainFiles), 'Test:', len(testFiles), 'Labels:', len(labels))


# # # print(labels.head())
# # # print(labels.describe())

# # imgs = np.array([np.array(Image.open(file).resize((224,224))) for file in trainFiles])

# # print('Shape:', imgs.shape)

# # ids = [file.split('/')[-1].split('.')[0] for file in trainFiles]
# # id_val = dict(zip(labels.id,labels.breed))
# # label_ids = [id_val[id] for id in ids]
# # label_index = {label:i for i,label in enumerate(np.unique(label_ids))}
# # labels_id = [label_index[label] for label in label_ids]

# # # pp.pprint(label_ids)
# # # pp.pprint(label_index)
# # # pp.pprint(labels_id)

# # labels_onehot_encoded = to_categorical(labels_id,num_classes=120)
# # base_model = VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))

# # model = Sequential(base_model.layers[:7])

# # model.add(Flatten())
# # model.add(Dense(256,activation='relu'))
# # #model.add(Dense(256,activation='relu'))
# # model.add(Dense(120,activation='softmax'))
# # model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

# # model.fit(imgs,labels_onehot_encoded,batch_size=8,validation_split=0.2, epochs=500)

# # model.save(modelPath)
