import os as os
import glob as glob
import shutil as shutil
import pandas as pd

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
    
    print('Creating directory:', dirname)
    os.mkdir(dirname)

def GetHomeDir():
    return str(Path.home())

def GetImageFilesInDir(inpPath):
    return glob.glob(os.path.join(inpPath, '*.jpg'))

def SetupKaggleData(srcPath, labelPath, outPath):    
    print('SetupKaggleData:', srcPath, labelPath, outPath)
    home = str(Path.home())
    tempPath = os.path.join(home, 'temp')
    print(tempPath)

    if not outPath.startswith(tempPath):
        print('Working folder should be in temp directory')
        return

    rmtree(outPath, ignore_errors=True)
    CreateDir(outPath)

    files = GetImageFilesInDir(srcPath)
    labels = pd.read_csv(labelPath)
    print(labels.describe())

    