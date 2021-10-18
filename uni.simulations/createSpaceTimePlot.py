import numpy
import vtk 
import os
import xml.etree.ElementTree as ElementTree
import meshio
from vtkmodules.vtkCommonCore import vtkDataArray
from vtk.util.numpy_support import vtk_to_numpy
import warnings
import numpy as np
import datetime
from shutil import copy
import matplotlib.pyplot as plt

# file stuff #
scriptFilePath = os.path.dirname(os.path.realpath(__file__))
scriptTimeStamp = str(datetime.datetime.now())
scriptTimeStamp = scriptTimeStamp.replace(":","-")
scriptTimeStamp = scriptTimeStamp.replace(" ","_")
# file stuff #





##### PARAMETERS #####
inputFolderPath =  scriptFilePath+"/../data/visualizeData/input/"
dataFolderPath = inputFolderPath+"/simulationData/"
dataFilePath = dataFolderPath+"u.pvd"
outputParentFolderPath = scriptFilePath + "/../data/visualizeData/output/"    #+timestamp -> kein parent mehr
STARTTIME = -1              
ENDTIME = -1            #-1 for all
SHOWNORMALIZED = False
##### PARAMETERS ##### 



### copy script to save it ###
outputFolderPath = outputParentFolderPath+scriptTimeStamp+"/"
os.mkdir(outputFolderPath)
copy(os.path.realpath(__file__), outputFolderPath+"used_script_visualization_"+scriptTimeStamp+".py")
### copy script to save it ###


### open pvd to get vtu data files
pvdXMLTree = ElementTree.parse(dataFilePath)
pvdXMLTreeRoot = pvdXMLTree.getroot()
infoString = "visualization info"
infoString += "\n\t"+"dataFilePath"+" = \t\t"+str(dataFilePath)

pythonFiles = 0
for file in os.listdir(str(dataFolderPath+"../")):
    if file.endswith(".py"):
        if pythonFiles == 0:
            pythonFilesInDir = file
        else:
            pythonFilesInDir += "\n"+file
        pythonFiles += 1
if pythonFiles != 1:
    warnings.warn(str("t\t\t"+str(pythonFiles)+" pyhton files in data directory ("+dataFolderPath+") gefunden"+"\nt\t\tnot copying any simulation script files\n"))
else:
    copy(str(dataFolderPath+"../"+pythonFilesInDir),outputFolderPath)
    
infoString += "\n\t"+"simulation script"+" = \t"+str(pythonFilesInDir)




# create empty array
timeFileArrayTemp = [["",""] for y in range(len(pvdXMLTreeRoot[0]))]

# load data file names
i = 0
for dataset in pvdXMLTreeRoot[0]:
    timeStep = float(dataset.get('timestep'))
    if timeStep>=STARTTIME and (timeStep<=ENDTIME or ENDTIME <0):
        timeFileArrayTemp[i][0]= timeStep
        timeFileArrayTemp[i][1]= dataset.get('file')
        i+=1
numberOfFiles = i

timeFileArray = [["",""] for y in range(numberOfFiles)]
timeFileArray = timeFileArrayTemp[0:numberOfFiles][:]


### read first vtu file to get mesh data 
data = meshio.read(dataFolderPath+timeFileArray[0][1])
Lmin=min(data.points[:,0])
Lmax=max(data.points[:,0])
L=Lmax-Lmin
infoString += "\n\t"+"L"+" = \t\t\t"+str(L)
cells=len(data.cells[0])
points=len(data.points[:,0])

infoString += "\n\t"+"points"+" = \t\t"+str(points)
n=cells
infoString += "\n\t"+"n"+" = \t\t\t"+str(n)
degree = round(points/cells)
infoString += "\n\t"+"degree"+" = \t\t"+str(degree)


# lock up all data file names
reader = vtk.vtkXMLUnstructuredGridReader()
filesNotFound = 0
filesFound = 0
listOfFilesFound = []
for i in range(numberOfFiles):
    if os.path.isfile(dataFolderPath+timeFileArray[i][1]):
        filesFound += 1
        listOfFilesFound.append(i)
    else:
        filesNotFound += 1

if filesNotFound>0:
    warnings.warn(str("\n\n\t\t\t"+str(filesNotFound)+" files from XML not found!\n"))
    print("\n")
    
    
# load all data files
dataArray = np.empty([filesFound, points])
t = np.empty(filesFound)
for i in range(filesFound):
    reader.SetFileName(dataFolderPath+timeFileArray[listOfFilesFound[i]][1])
    reader.Update()
    readerOutput = reader.GetOutput()
    vtkArray = readerOutput.GetPointData().GetArray(0)
        #https://stackoverflow.com/a/54072929
        #entweder index (-> 0) oder den namen in <PointData Vectors="function_8[0]">     -> function_8[0]
    multiComponentArray = vtk_to_numpy(vtkArray)
    dataArray[i,:] = multiComponentArray[:,0]
    t[i] = timeFileArray[listOfFilesFound[i]][0]
functionName = vtkArray.GetName()
infoString += "\n\t"+"functionName"+" = \t\t"+str(functionName)

# create normalized data array
dataArrayNormalized = numpy.copy(dataArray)
LinftyDataArray = 0
for i in range(filesFound):
 if max(abs(dataArrayNormalized[i,:]))> LinftyDataArray:
     LinftyDataArray = max(abs(dataArrayNormalized[i,:]))
infoString += "\n\t"+"LinftyDataArray"+" = \t"+str(LinftyDataArray)

for i in range(filesFound):
    dataArrayNormalized[i,:] = LinftyDataArray/max(abs(dataArrayNormalized[i,:]))*np.copy(dataArrayNormalized[i,:])


### plotting ###

# generate 2 2d grids for the x & y bounds
x = np.linspace(Lmin, Lmax, points)


COLORRESOLUTION = 100
NUMBEROFCOLORTICKS = 9

fig = plt.figure()
fig.suptitle(pythonFilesInDir)
levels = np.linspace(-LinftyDataArray,LinftyDataArray,COLORRESOLUTION+1)
if SHOWNORMALIZED:
    dataAx = fig.add_subplot(121)
else:
    dataAx = fig.add_subplot(111)
    
dataContourf = dataAx.contourf(x,t,dataArray, levels=levels, cmap='coolwarm')
if SHOWNORMALIZED:
    dataNormalizedAx = fig.add_subplot(122)
    dataNormalizedContourf = dataNormalizedAx.contourf(x,t,dataArrayNormalized, levels=101, cmap='coolwarm')
    colorbar = fig.colorbar(dataContourf,ax=[dataAx,dataNormalizedAx], location='right')        #werte von dataContourf, aber an beiden ([dataAx,dataNormalizedAx]) damit kein ax kleiner wird
    
    dataNormalizedAx.set_title("normalized")    
    dataNormalizedAx.set_xlabel("x")
    #dataNormalizedAx.yaxis.set_label_position('right')
    dataNormalizedAx.set_yticks([])  
    dataNormalizedAx.yaxis.tick_right()

else:
    colorbar = fig.colorbar(dataContourf,ax=[dataAx], location='right')        #werte von dataContourf, aber an beiden ([dataAx,dataNormalizedAx]) damit kein ax kleiner wird
ticksLinSpace = np.linspace(-LinftyDataArray,LinftyDataArray,NUMBEROFCOLORTICKS)
colorbar.set_ticks(ticksLinSpace)
dataAx.set_title(functionName)
dataAx.set_xlabel("x")
dataAx.set_ylabel("t")




### write info to file

print(infoString)
copy(inputFolderPath+"info.txt", outputFolderPath+"info.txt")
infoFile = open(outputFolderPath+"info.txt","a")
infoFile.write("\n\n")
infoFile.write(infoString)
infoFile.close()

### show plot
plt.show()



