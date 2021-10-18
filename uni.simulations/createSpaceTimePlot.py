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
from meshio import _files

scriptStartTime = datetime.datetime.now()
print(scriptStartTime,"starting ...")  
# file stuff #
scriptFilePath = os.path.dirname(os.path.realpath(__file__))
scriptTimeStamp = str(datetime.datetime.now())
scriptTimeStamp = scriptTimeStamp.replace(":","-")
scriptTimeStamp = scriptTimeStamp.replace(" ","_")
# file stuff #





##### PARAMETERS #####
TITLE = "kappa = 1 (pure Kuramoto-Sivashinsky)"
inputFolderPath =  scriptFilePath+"/../data/visualizeData/input/"
dataFolderPath = inputFolderPath+"simulationData/"
dataFilePath = dataFolderPath+"u.pvd"
outputParentFolderPath = scriptFilePath + "/../data/visualizeData/output/"    #+timestamp -> kein parent mehr
STARTTIME = -1              
ENDTIME = 1            #-1 for all
SHOWNORMALIZED = True
##### PARAMETERS ##### 



### copy script to save it ###
titleNameReplacedCharacters = TITLE.replace(" ","_")
titleNameReplacedCharacters = titleNameReplacedCharacters.replace("/",":")
outputFolderPath = outputParentFolderPath+titleNameReplacedCharacters+"_"+scriptTimeStamp+"/"
os.mkdir(outputFolderPath)
usedVisualizationScriptName = "used_script_visualization_"+scriptTimeStamp+".py"
copy(os.path.realpath(__file__), outputFolderPath+usedVisualizationScriptName)
### copy script to save it ###


### open pvd to get vtu data files
pvdXMLTree = ElementTree.parse(dataFilePath)
pvdXMLTreeRoot = pvdXMLTree.getroot()
infoString = "visualization info"
infoString += "\n\t"+"dataFilePath"+" = \t\t\t"+str(dataFilePath)

pythonFiles = 0
searchPaths = [dataFolderPath+"../",dataFolderPath]
pythonFileName = ""
for searchPath in searchPaths:
    for file in os.listdir(searchPath):
        if file.endswith(".py"):
            pythonFileName += file
            if pythonFiles == 0:
                pythonFilesInDirPaths = searchPath+file
            else:
                pythonFilesInDirPaths += "\n"+searchPath+file
            pythonFiles += 1
if pythonFiles != 1:
    warnings.warn(str("\t\t\t"+str(pythonFiles)+" pyhton files in data directory ("+str(dataFolderPath+"/..")+" and "+dataFolderPath+") gefunden"+"\nt\t\tnot copying any simulation script files\n"))
else:
    copy(pythonFilesInDirPaths,outputFolderPath)
    
infoString += "\n\t"+"simulation script"+" = \t\t"+str(pythonFilesInDirPaths)




# create empty array
timeFileArrayTemp = [["",""] for y in range(len(pvdXMLTreeRoot[0]))]

print(datetime.datetime.now(),"loading data file names")  
# load data file names
i = 0
logEveryXtimes = 1000
filesSinceLog = 0
logEveryXpercent = 10
nextLogPercentage = logEveryXpercent
for dataset in pvdXMLTreeRoot[0]:
    timeStep = float(dataset.get('timestep'))
    if timeStep>=STARTTIME and (timeStep<=ENDTIME or ENDTIME <0):
        timeFileArrayTemp[i][0]= timeStep
        timeFileArrayTemp[i][1]= dataset.get('file')
        i+=1
    if i/len(pvdXMLTreeRoot[0]) > nextLogPercentage/100:
#        print(datetime.datetime.now(),"loaded "+str(round(i/len(pvdXMLTreeRoot[0])*100))+"% of data file names")
        nextLogPercentage += logEveryXpercent
print(datetime.datetime.now(),"loaded 100% of data file names")
numberOfFiles = i

timeFileArray = [["",""] for y in range(numberOfFiles)]
timeFileArray = timeFileArrayTemp[0:numberOfFiles][:]


### read first vtu file to get mesh data 
print(datetime.datetime.now(),"loading first file for mesh data") 
data = meshio.read(dataFolderPath+timeFileArray[0][1])
Lmin=min(data.points[:,0])
Lmax=max(data.points[:,0])
L=Lmax-Lmin
infoString += "\n\t"+"L"+" = \t\t\t\t"+str(L)
cells=len(data.cells[0])
points=len(data.points[:,0])

infoString += "\n\t"+"points"+" = \t\t\t"+str(points)
n=cells
infoString += "\n\t"+"n"+" = \t\t\t\t"+str(n)


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
print(datetime.datetime.now(),"loading data files") 
dataArray = np.empty([filesFound, points])
t = np.empty(filesFound)
logEveryXpercent = 10
nextLogPercentage = logEveryXpercent
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
    if i/filesFound > nextLogPercentage/100:
        print(datetime.datetime.now(),"loaded "+str(round(i/filesFound*100))+"% of data files")
        nextLogPercentage += logEveryXpercent
print(datetime.datetime.now(),"loaded 100% of data files")
        
functionName = vtkArray.GetName()
infoString += "\n\t"+"functionName"+" = \t\t\t"+str(functionName)

print(datetime.datetime.now(),"creating normalized array") 
# create normalized data array
dataArrayNormalized = numpy.copy(dataArray)
LinftyDataArray = 0
for i in range(filesFound):
    if max(abs(dataArrayNormalized[i,:]))> LinftyDataArray:
        LinftyDataArray = max(abs(dataArrayNormalized[i,:]))
infoString += "\n\t"+"LinftyDataArray"+" = \t\t"+str(LinftyDataArray)

for i in range(filesFound):
    dataArrayNormalized[i,:] = LinftyDataArray/max(abs(dataArrayNormalized[i,:]))*np.copy(dataArrayNormalized[i,:])


### plotting ###

# generate 2 2d grids for the x & y bounds
x = np.linspace(Lmin, Lmax, points)


COLORRESOLUTION = 100
NUMBEROFCOLORTICKS = 9

print(datetime.datetime.now(),"creating plot") 
fig = plt.figure()
if SHOWNORMALIZED:
    fig.set_size_inches(16, 9)
else:
    fig.set_size_inches(10, 9)
    
plt.suptitle(TITLE)
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

fig.text(0.1, 0,"simulation script    " + pythonFileName + "\n"+"visualization script " +usedVisualizationScriptName, size=10, ha='left', va='bottom')




### write info to file

print(infoString)
originalInfoFilePath = inputFolderPath+"info.txt"
if os.path.isfile(originalInfoFilePath):
    copy(originalInfoFilePath, outputFolderPath+"info.txt")
infoFile = open(outputFolderPath+"info.txt","a")
scriptEndTime = datetime.datetime.now()
scriptTime = scriptEndTime-scriptStartTime
print(scriptEndTime,"finishing after "+str(scriptTime))  
infoString += "\n\t"+"total visualization time"+" = \t"+str(scriptTime)
infoFile.write("\n\n")
infoFile.write(infoString)
infoFile.close()

### show/export plot
print(datetime.datetime.now(),"plotting")
plt.savefig(outputFolderPath+"plot.png", dpi = 128, bbox_inches='tight')
print(datetime.datetime.now(),"exported")
#plt.show()


print(datetime.datetime.now(),"finished")


