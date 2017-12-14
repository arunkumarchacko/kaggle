import os as os
import glob as glob
import shutil as shutil
import pandas as pd
import pprint as pp
import random as random

from pathlib import Path
from shutil import rmtree, copyfile


def CopyFile(start, end, pattern, inpDir, outDir, extn):
    print('CopyFile:', (start, end, pattern, inpDir, outDir, extn))
    files = sorted([f for f in glob.glob(os.path.join(inpDir, extn)) if pattern in f])
    # print(files)

    for f in files[start:end]:
        outPath = os.path.join(outDir, os.path.basename(f))
        print('Copying', f, outPath)
        shutil.copyfile(f, outPath)

def CreateDir(dirname):
    if os.path.exists(dirname):
        return
    
    # print('Creating directory:', dirname)
    os.mkdir(dirname)

def GetHomeDir():
    return str(Path.home())

def GetImageFilesInDir(inpPath):
    return [f for f in glob.iglob(os.path.join(inpPath, '**/*.jpg'), recursive=True)]

def CreateCategoryDirectories(labels, outDir, colName):
    for category in set(labels[colName].tolist()):
        CreateDir(os.path.join(outDir, category))

def SetupKaggleData(srcPath, labelPath, outDir, colName, origTestDir):    
    print('SetupKaggleData:', srcPath, labelPath, outDir)
    home = str(Path.home())
    tempPath = os.path.join(home, 'temp')
    print(tempPath)

    # if os.path.exists(outDir):
    #     return

    if not outDir.startswith(tempPath):
        print('Working folder should be in temp directory', outDir)
        return

    print('Removing', outDir)
    rmtree(outDir, ignore_errors=False)
    CreateDir(outDir)

    labels = pd.read_csv(labelPath)
    # print(labels.describe())

    trainDir = os.path.join(outDir, 'train')
    validDir = os.path.join(outDir, 'valid')
    trialDir = os.path.join(outDir, 'trial')
    testDir = os.path.join(os.path.join(outDir, 'test_root'), 'test')
    testTrialDir = os.path.join(os.path.join(outDir, 'test_trial_root'), 'test')

    CreateDir(trainDir)
    CreateDir(validDir)
    CreateDir(trialDir)
    CreateDir(os.path.join(outDir, 'test_root'))
    CreateDir(testDir)
    CreateDir(os.path.join(outDir, 'test_trial_root'))
    CreateDir(testTrialDir)

    CreateCategoryDirectories(labels, trainDir, colName)
    CreateCategoryDirectories(labels, validDir, colName)
    CreateCategoryDirectories(labels, trialDir, colName)

    files = GetImageFilesInDir(srcPath)

    print('Found {} files in directory {}'.format(len(files), srcPath))

    random.shuffle(files)
    imgToBreed = {}
    for index, row in labels.iterrows():    
        imgToBreed[row['id']] = row[colName]

    trainCount = int(len(files) * .9)
    tCount = 0
    for f in  files[:trainCount]:
        imgId = os.path.basename(f).split('.')[0]        
        cpPath = os.path.join(os.path.join(trainDir, imgToBreed[imgId]), os.path.basename(f))
        shutil.copyfile(f, cpPath)
        tCount+=1

        if tCount < 1000:
            cpPath = os.path.join(os.path.join(trialDir, imgToBreed[imgId]), os.path.basename(f))
            shutil.copyfile(f, cpPath)

    for f in  files[trainCount:]:
        imgId = os.path.basename(f).split('.')[0]
        cpPath = os.path.join(os.path.join(validDir, imgToBreed[imgId]), os.path.basename(f))
        shutil.copyfile(f, cpPath)
    
    print('TrainCount', len(GetImageFilesInDir(trainDir)))
    print('ValidCount', len(GetImageFilesInDir(validDir)))
    print('TrialCount', len(GetImageFilesInDir(trialDir)))

    files = GetImageFilesInDir(origTestDir)
    print('Found {} test items in {}'.format(len(files), origTestDir))
    for f in files:
        cpPath = os.path.join(os.path.join(testDir, os.path.basename(f)))
        shutil.copyfile(f, cpPath)
    

    for f in files[:20]:
        cpPath = os.path.join(os.path.join(testTrialDir, os.path.basename(f)))
        shutil.copyfile(f, cpPath)
    


    