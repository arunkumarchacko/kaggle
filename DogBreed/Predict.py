
import sys
sys.path.append('../')
import utils as utils

import pickle as pickle
import pandas as pd
import os
import numpy as np
import pprint as pp
import keras
from keras.preprocessing.image import ImageDataGenerator
import random as random


home = utils.GetHomeDir()
modelPath = os.path.join(home, 'ml/data/kaggle/dogbreed/model2.h5') 
featuresPath = os.path.join(home, 'ml/data/kaggle/dogbreed/features.pkl')
submissionFilePath = os.path.join(home, 'ml/data/kaggle/dogbreed/kaggle_submission1.csv')
# testDir = os.path.join(home, 'ml/data/kaggle/dogbreed/test/')

testDir = os.path.join(home, 'temp/dog_breed_workingdir/test_trial_root')
# testDir = os.path.join(home, 'temp/dog_breed_workingdir/test_root')

print(testDir)
model = keras.models.load_model(modelPath)
# print(model.summary())

img_width, img_height = 224, 224

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(testDir, target_size=(img_height, img_width), shuffle = "false", class_mode='categorical', batch_size=1)
# test_generator = test_datagen.flow_from_directory(trainDir, target_size=(img_height, img_width), batch_size=1, class_mode='categorical')

filenames = test_generator.filenames
nb_samples = len(filenames)

print(nb_samples)

with open(featuresPath, 'rb') as handle:
    b = pickle.load(handle)

cols = [k for k in b.keys()]
cols.insert(0, 'id')

df = pd.DataFrame(columns=cols)

predict = model.predict_generator(test_generator, steps = nb_samples, verbose=1)

for i,p in enumerate(predict):
    row = [pred for pred in p]    
    row.insert(0, os.path.basename(filenames[i]).split('.')[0])    
    df.loc[i] = row
    print(i, np.argmax(p), p[np.argmax(p)], np.sum(p))
        
print('Writing to', submissionFilePath)
df.to_csv(submissionFilePath, index=False)
