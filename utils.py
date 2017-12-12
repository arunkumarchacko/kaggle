import os as os
import glob as glob
import shutil as shutil

def CopyFile(start, end, pattern, inpDir, outDir, extn):
    print('CopyFile:', (start, end, pattern, inpDir, outDir, extn))
    files = sorted([f for f in glob.glob(os.path.join(inpDir, extn)) if pattern in f])
    # print(files)

    for f in files[start:end]:
        outPath = os.path.join(outDir, os.path.basename(f))
        print('Copying', f, outPath)
        shutil.copyfile(f, outPath)