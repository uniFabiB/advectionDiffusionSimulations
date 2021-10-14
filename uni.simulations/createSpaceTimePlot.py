import numpy
import vtk 
import os
import xml.etree.ElementTree as ElementTree
import meshio
from vtkmodules.vtkCommonCore import vtkDataArray
from vtk.util.numpy_support import vtk_to_numpy
import warnings
import numpy as np

# filestuff #
scriptFilePath = os.path.dirname(os.path.realpath(__file__))
# filestuff #





##### PARAMETERS ##### 
dataFolderPath = scriptFilePath+"/../data/loadData/"
dataFilePath = dataFolderPath+"u.pvd"
STARTTIME = -1              
ENDTIME = -1            #-1 for all
SHOWNORMALIZED = False
##### PARAMETERS ##### 



pvdXMLTree = ElementTree.parse(dataFilePath)
pvdXMLTreeRoot = pvdXMLTree.getroot()
infoString = "infostring"
infoString += "\n\t"+"dataFilePath"+" = \t\t"+str(dataFilePath)

pythonFiles = 0
for file in os.listdir(dataFolderPath):
    if file.endswith(".py"):
        if pythonFiles == 0:
            pythonFilesInDir = file
        else:
            pythonFilesInDir += "\n"+file
        pythonFiles += 1
if pythonFiles != 1:
    warnings.warn(str("t\t\t"+str(pythonFiles)+" pyhton files in data directory ("+dataFolderPath+") gefunden"+"\n"))
infoString += "\n\t"+"pythonFilesInDir"+" = \t"+str(pythonFilesInDir)




## create empty array
timeFileArrayTemp = [["",""] for y in range(len(pvdXMLTreeRoot[0]))]


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

data = meshio.read(dataFolderPath+timeFileArray[0][1])
Lmin=min(data.points[:,0])
Lmax=max(data.points[:,0])
L=Lmax-Lmin
#print(data)
infoString += "\n\t"+"L"+" = \t\t\t"+str(L)
cells=len(data.cells[0])
points=len(data.points[:,0])

infoString += "\n\t"+"points"+" = \t\t"+str(points)
n=cells
infoString += "\n\t"+"n"+" = \t\t\t"+str(n)
degree = round(points/cells)
infoString += "\n\t"+"degree"+" = \t\t"+str(degree)

#https://stackoverflow.com/a/54072929

# load a vtk file as input
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
    
    
    
dataArray = np.empty([filesFound, points])
t = np.empty(filesFound)
for i in range(filesFound):
    reader.SetFileName(dataFolderPath+timeFileArray[listOfFilesFound[i]][1])
    reader.Update()
    readerOutput = reader.GetOutput()
    vtkArray = readerOutput.GetPointData().GetArray(0)
    multiComponentArray = vtk_to_numpy(vtkArray)
    dataArray[i,:] = multiComponentArray[:,0]
    #dataArray[i,:] = np.random.rand(points)            just to test if the values match
    t[i] = timeFileArray[listOfFilesFound[i]][0]
functionName = vtkArray.GetName()
infoString += "\n\t"+"functionName"+" = \t\t"+str(functionName)


dataArrayNormalized = numpy.copy(dataArray)
LinftyDataArray = 0
for i in range(filesFound):
 if max(abs(dataArrayNormalized[i,:]))> LinftyDataArray:
     LinftyDataArray = max(abs(dataArrayNormalized[i,:]))
infoString += "\n\t"+"LinftyDataArray"+" = \t"+str(LinftyDataArray)


for i in range(filesFound):
    dataArrayNormalized[i,:] = LinftyDataArray/max(abs(dataArrayNormalized[i,:]))*np.copy(dataArrayNormalized[i,:])

#entweder index (-> 0) oder den namen in <PointData Vectors="function_8[0]">     -> function_8[0]
import matplotlib.pyplot as plt

# generate 2 2d grids for the x & y bounds
x = np.linspace(Lmin, Lmax, points)


COLORRESOLUTION = 100

fig = plt.figure()
fig.suptitle(pythonFilesInDir)
if SHOWNORMALIZED:
    dataAx = fig.add_subplot(121)
else:
    dataAx = fig.add_subplot(111)
    
dataContourf = dataAx.contourf(x,t,dataArray, COLORRESOLUTION, cmap='coolwarm')
if SHOWNORMALIZED:
    dataNormalizedAx = fig.add_subplot(122)
    dataNormalizedContourf = dataNormalizedAx.contourf(x,t,dataArrayNormalized, COLORRESOLUTION, cmap='coolwarm')
    fig.colorbar(dataContourf,ax=[dataAx,dataNormalizedAx], location='right')        #werte von dataContourf, aber an beiden ([dataAx,dataNormalizedAx]) damit kein ax kleiner wird
    
    dataNormalizedAx.set_title("normalized")    
    dataNormalizedAx.set_xlabel("x")
    #dataNormalizedAx.yaxis.set_label_position('right')
    dataNormalizedAx.set_yticks([])  
    dataNormalizedAx.yaxis.tick_right()

else:
    fig.colorbar(dataContourf,ax=[dataAx], location='right')        #werte von dataContourf, aber an beiden ([dataAx,dataNormalizedAx]) damit kein ax kleiner wird
    
dataAx.set_title(functionName)
dataAx.set_xlabel("x")
dataAx.set_ylabel("t")


    

print(infoString)

plt.show()


#fig, axes = pl.subplots(1, 3, figsize=(15, 4))
#c1 = axes[0].contourf(x, t, dataArray)

#plt.colorbar(c1, ax=axes[0])
#plt.imshow(x,t,dataArray, cmap='RdBu', aspect='auto')
#plt.show()



