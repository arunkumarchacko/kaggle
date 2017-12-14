
import sys
sys.path.append('../')
import utils as utils

import os
import pprint as pp
import keras
from keras.preprocessing.image import ImageDataGenerator


home = utils.GetHomeDir()
modelPath = os.path.join(home, 'ml/data/kaggle/dogbreed/model2.h5') 
# testDir = os.path.join(home, 'ml/data/kaggle/dogbreed/test/')

testDir = os.path.join(home, 'temp/dog_breed_workingdir/test_trial_root')

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

# predict = model.predict_generator(test_generator,steps = 10)
predict = model.predict_generator(test_generator, steps = nb_samples, verbose=1)

print(predict)