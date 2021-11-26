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
inputFolderPath =  scriptFilePath+"/../data/exportNumpyArray/input/"
dataFolderPath = inputFolderPath+"simulationData/"
dataFilePath = dataFolderPath+"u.pvd"
outputParentFolderPath = scriptFilePath + "/../data/exportNumpyArray/output/"    #+timestamp -> kein parent mehr
timestepToExport=2000
##### PARAMETERS ##### 



### copy script to save it ###
outputFolderPath = outputParentFolderPath+scriptTimeStamp+"/"
os.mkdir(outputFolderPath)
usedVisualizationScriptName = "used_script_visualization_"+scriptTimeStamp+".py"
copy(os.path.realpath(__file__), outputFolderPath+usedVisualizationScriptName)
### copy script to save it ###


### open pvd to get vtu data files
pvdXMLTree = ElementTree.parse(dataFilePath)
pvdXMLTreeRoot = pvdXMLTree.getroot()
infoString = "numpy export info"
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
    raise Exception(str("\t\t\t"+str(pythonFiles)+" pyhton files in data directory ("+str(dataFolderPath+"/..")+" and "+dataFolderPath+") gefunden"+"\nt\t\tnot copying any simulation script files\n"))
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
    #multiComponentArray only if vector valued function
    #multiComponentArray = vtk_to_numpy(vtkArray)
    #dataArray[i,:] = multiComponentArray[:,0]
    dataArray[i,:] = vtk_to_numpy(vtkArray)
    t[i] = timeFileArray[listOfFilesFound[i]][0]
    if i/filesFound > nextLogPercentage/100:
        print(datetime.datetime.now(),"loaded "+str(round(i/filesFound*100))+"% of data files")
        nextLogPercentage += logEveryXpercent
print(datetime.datetime.now(),"loaded 100% of data files")
        
functionName = vtkArray.GetName()
infoString += "\n\t"+"functionName"+" = \t\t\t"+str(functionName)

# finding the best fitting t
bestT = t[0]
bestI = 0
for i in range(filesFound):
    if abs(t[i]-timestepToExport)<abs(bestT - timestepToExport):
        bestT = t[i]
        bestI = i
        
        
infoString += "\n\t"+"timestepToExport"+" = \t\t\t"+str(timestepToExport)
infoString += "\n\t"+"bestT"+" = \t\t\t"+str(bestT)
infoString += "\n\t"+"bestI"+" = \t\t\t"+str(bestI)

# export timestep bestT
dataArrayBestTime = dataArray[bestI,:]
# values are twice (by difference cells and points) so e.g. -0.96056498 -0.96056498 -0.8944293 -0.8944293 ...
#OutputArray = np.zeros(n)
#for i in range(round(n/2)):
#    OutputArray[i] = dataArrayBestTime[4*i]
#    OutputArray[i+round(n/2)] = dataArrayBestTime[4*i]
for i in range(n):
    OutputArray[i] = dataArrayBestTime[2*i]
 


scriptTimeStamp = str(datetime.datetime.now())
scriptTimeStamp = scriptTimeStamp.replace(":","-")
scriptTimeStamp = scriptTimeStamp.replace(" ","_")

print(OutputArray)
np.save(outputFolderPath +functionName+"_t-"+str(bestI)+"_"+scriptTimeStamp, OutputArray)


### write info to file
print(infoString)
originalInfoFilePath = inputFolderPath+"info.txt"
if os.path.isfile(originalInfoFilePath):
    copy(originalInfoFilePath, outputFolderPath+"info.txt")
infoFile = open(outputFolderPath+"info.txt","a")
scriptEndTime = datetime.datetime.now()
scriptTime = scriptEndTime-scriptStartTime
print(scriptEndTime,"total time"+" = \t"+str(scriptTime))
infoString += "\n\t"+"total time"+" = \t"+str(scriptTime)
infoFile.write("\n\n")
infoFile.write(infoString)
infoFile.close()
#plt.show()


print(datetime.datetime.now(),"finished")


