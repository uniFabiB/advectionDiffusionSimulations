import numpy as np
import cv2
import os

scriptFilePath = os.path.dirname(os.path.realpath(__file__))

# if error importing cv2 > terminal 

inputFolderPath =  scriptFilePath+"/../data/0combinePngFiles/input/"
ouputFolderPath =  scriptFilePath+"/../data/0combinePngFiles/output/"

files = []
filesInDirectory = os.listdir(inputFolderPath)
filesInDirectory.sort()
for file in filesInDirectory:
    if file.endswith(".png"):
        files.append(file)


infoImage = cv2.imread(inputFolderPath+files[0])
width = len(infoImage[0])
height = len(infoImage)


images = []

blankImage = np.ones((height,300,3), np.uint8)
blankImage = 225*blankImage

for file in files:
    images.append(cv2.imread(inputFolderPath+file))
    images.append(blankImage)

    

combinedImage = np.concatenate(images[:-1], axis=1)      # axis = 1 horizontal, 0 = vertical
cv2.imwrite(ouputFolderPath+'out.png', combinedImage)
print("finished")