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
TITLE = "L=REPLACElBYSCRIPT, kappa=REPLACEkappaBYSCRIPT, epsilon=REPLACEepsilonSBYSCRIPT, ic=REPLACEinitialconditionBYSCRIPT"
inputFolderPath =  scriptFilePath+"/../data/0visualizeData/input/"
dataFolderPath = inputFolderPath+"simulationData/"
dataFilePath = dataFolderPath+"u.pvd"
timeDataFilePath = dataFolderPath+"timeFunctions_0.vtu"
originalInfoFilePath = inputFolderPath+"info.txt"
outputParentFolderPath = scriptFilePath + "/../data/0visualizeData/output/"    #+timestamp -> kein parent mehr
STARTTIME = -1
ENDTIME = -1 #STARTTIME+200            #-1 for all
SHOWNORMALIZED = False
FREEZSOLUTION = True
##### PARAMETERS #####
DEBUGGING = False


### copy script to save it ###
    
infoFile = open(originalInfoFilePath,"r")
infoFileLines = infoFile.readlines()
infoFile.close()
def getInfoFromInfoFile(infoFileLines, searchString):
    ## should include \tL sonst nProL -> wrong
    result = "undefined"
    endString = "\n"
    for line in infoFileLines:
        if line.find(searchString)>-1:
            indexStart = line.index(searchString);
            indexEnd = line.index(endString);
            result = line[indexStart+len(searchString):indexEnd]
    if result == "undefined":
        warnings.warn(str(searchString + "not found in infoFileLines"))
    result = result.replace(" ","")
    result = result.replace("\t","")
    print("searching for "+"searchString" + " in infoFile results in " + result)
    return result
    

L = getInfoFromInfoFile(infoFileLines, "\tL =")
kappa = getInfoFromInfoFile(infoFileLines, "\tkappa =")
epsilon = getInfoFromInfoFile(infoFileLines, "\tepsilonKdVKuraSiva =")
initialCondition = getInfoFromInfoFile(infoFileLines, "\tinitialCondition =")



TITLE = TITLE.replace("REPLACElBYSCRIPT", str(L))
TITLE = TITLE.replace("REPLACEkappaBYSCRIPT", str(kappa))
TITLE = TITLE.replace("REPLACEepsilonSBYSCRIPT", str(epsilon))
TITLE = TITLE.replace("REPLACEinitialconditionBYSCRIPT", str(initialCondition))

titleNameReplacedCharacters = TITLE.replace(" ","_")
titleNameReplacedCharacters = titleNameReplacedCharacters.replace("/",":")
outputFolderPath = outputParentFolderPath+scriptTimeStamp+"_"+titleNameReplacedCharacters+"/"
usedVisualizationScriptName = "used_script_visualization_"+scriptTimeStamp+".py"
if not(DEBUGGING):
    os.mkdir(outputFolderPath)
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
    raise Exception(str("\t\t\t"+str(pythonFiles)+" pyhton files in data directory ("+str(dataFolderPath+"/..")+" and "+dataFolderPath+") gefunden"+"\nt\t\tnot copying any simulation script files\n"))
else:
    if not(DEBUGGING):
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
if len(timeFileArray) == 0:
    raise Exception("no matching files found for this time range")
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
    dataTemp = vtk_to_numpy(vtkArray)
    dim = dataTemp.ndim
    if dim == 1:
        dataArray[i,:] = dataTemp
    else:
        dataArray[i,:] = dataTemp[:,0]
    t[i] = timeFileArray[listOfFilesFound[i]][0]
    if i/filesFound > nextLogPercentage/100:
        print(datetime.datetime.now(),"loaded "+str(round(i/filesFound*100))+"% of data files")
        nextLogPercentage += logEveryXpercent
print(datetime.datetime.now(),"loaded 100% of data files")
if dim != 1:
    warnings.warn("only 0th (1st) component of vector valued function used")
    
# load time data file
timeFunctionsReader = vtk.vtkXMLUnstructuredGridReader()
timeFunctionsReader.SetFileName(timeDataFilePath)
timeFunctionsReader.Update()
timeFunctionsReaderOutput = timeFunctionsReader.GetOutput()




timeFunctionsTimeVtkArray = timeFunctionsReaderOutput.GetPointData().GetArray("time")
# create timeFunctionsTime from spatial functions in case no timefunctino "time" is given (old simulations)
#spatialTimeUnfiltered = np.zeros(len(pvdXMLTreeRoot[0]))
#for i in range(len(spatialTimeUnfiltered)):
#    if timeFileArrayTemp[i][0]== "":
#        spatialTimeUnfiltered[i] = -99999
#    else:
#        spatialTimeUnfiltered[i] = timeFileArrayTemp[i][0]
#timeFunctionsTimeUnfiltered = spatialTimeUnfiltered
timeFunctionsTimeUnfiltered = vtk_to_numpy(timeFunctionsTimeVtkArray)




kdvSpeedVtkArray = timeFunctionsReaderOutput.GetPointData().GetArray("kdv speed")

kdvSpeedUnfiltered = vtk_to_numpy(kdvSpeedVtkArray)

# calculate shift
kdvSpaceTraveledUnfiltered = np.zeros(len(timeFunctionsTimeUnfiltered))
kdvSpaceTraveledUnfiltered[0] = 0
for i in range(len(timeFunctionsTimeUnfiltered)-1):
    kdvSpaceTraveledUnfiltered[i+1] = kdvSpaceTraveledUnfiltered[i] + (timeFunctionsTimeUnfiltered[i+1]-timeFunctionsTimeUnfiltered[i])*kdvSpeedUnfiltered[i]
    
def filterTimeFunctionForSpecifiedTimeInterval(unfilteredTimeArray, unfilteredFunctionArray, tmin, tmax):
    index = 0
    funcTemp = np.zeros(len(unfilteredTimeArray))
    for i in range(len(unfilteredTimeArray)):
        timeStep = unfilteredTimeArray[i]
        if timeStep>=tmin and (timeStep<=tmax or tmax <0):
            funcTemp[index] = unfilteredFunctionArray[i]
            index += 1
    return funcTemp[:index]
    
# filter timeFunctions for specified time interval
timeFunctionsTime = filterTimeFunctionForSpecifiedTimeInterval(timeFunctionsTimeUnfiltered, timeFunctionsTimeUnfiltered, STARTTIME, ENDTIME)
timeFunctionsSpaceTraveled = filterTimeFunctionForSpecifiedTimeInterval(timeFunctionsTimeUnfiltered, kdvSpaceTraveledUnfiltered, STARTTIME, ENDTIME)
timeFunctionsSpeed = filterTimeFunctionForSpecifiedTimeInterval(timeFunctionsTimeUnfiltered, kdvSpeedUnfiltered, STARTTIME, ENDTIME)


# check time of timeFunctions matches time of spatial functions
for i in range(min(len(timeFunctionsTime),len(t))):
    if timeFunctionsTime[i] != t[i]:
        print("time functions of space functions and time functions dont match in \t i=",i,"th position","\n\t"+"space functions time ",t,"\n\t","time functions time ",timeFunctionsTime)
        warnings.warn("time functions of space functions and time functions dont match"+"\n\t"+"space functions time "+t+"\n\t"+"time functions time "+timeFunctionsTime)


functionName = vtkArray.GetName()
infoString += "\n\t"+"functionName"+" = \t\t\t"+str(functionName)


freezeDataArray = np.empty([filesFound, points])
if FREEZSOLUTION:
    print(datetime.datetime.now(),"creating freezing array") 
#if False:
    spatialIndizes = len(dataArray[0,:])
    spaceStep = L/spatialIndizes
    for i in range(len(dataArray[:,0])):
        indexShift = round(timeFunctionsSpaceTraveled[i]/spaceStep)  % spatialIndizes
        freezeDataArray[i,:] = np.roll(dataArray[i,:],-indexShift)


# create normalized data array
dataArrayNormalized = numpy.copy(dataArray)

LinftyDataArray = 0
for i in range(filesFound):
    if max(abs(dataArrayNormalized[i,:]))> LinftyDataArray:
        LinftyDataArray = max(abs(dataArrayNormalized[i,:]))
        infoString += "\n\t"+"LinftyDataArray"+" = \t\t"+str(LinftyDataArray)    
if SHOWNORMALIZED:
    print(datetime.datetime.now(),"creating normalized array") 
    for i in range(filesFound):
        dataArrayNormalized[i,:] = LinftyDataArray/max(abs(dataArrayNormalized[i,:]))*np.copy(dataArrayNormalized[i,:])
        
        
        

### plotting ###

# generate 2 2d grids for the x & y bounds
x = np.linspace(Lmin, Lmax, points)


COLORRESOLUTION = 100
NUMBEROFCOLORTICKS = 9

print(datetime.datetime.now(),"assigning plot") 
# default figsize [6.4, 4.8]
additionalFigHeight = 0
if ENDTIME < 0:
    actualEndTime = max(t)
else:
    actualEndTime = ENDTIME
if STARTTIME < 0:
    actualStartTime = min(t)
else:
    actualStartTime = STARTTIME
if actualEndTime-actualStartTime > 1000: 
    additionalFigHeight = (actualEndTime-actualStartTime - 1000)/1000*3/4*9 # factor 3/4 because the actual plot takes about 3/4 of the whole picture
fig = plt.figure()

figWidht = 10
numberOfPlots = 1
if SHOWNORMALIZED:
    figWidht += 6
    numberOfPlots += 1
if FREEZSOLUTION:
    figWidht += 6
    numberOfPlots += 1
fig.set_size_inches(figWidht, 9+additionalFigHeight)
    
plt.suptitle(TITLE)
levels = np.linspace(-LinftyDataArray,LinftyDataArray,COLORRESOLUTION+1)

plotNumber = 1
dataAx = fig.add_subplot(1,numberOfPlots,plotNumber)


dataContourf = dataAx.contourf(x,t,dataArray, levels=levels, cmap='coolwarm')

axis = [dataAx]

if FREEZSOLUTION:
    plotNumber += 1
    dataFreezeAx = fig.add_subplot(1,numberOfPlots,plotNumber)
    dataFreezeContourf = dataFreezeAx.contourf(x,t,freezeDataArray, levels=levels, cmap='coolwarm')
    
    dataFreezeAx.set_title("frozen")    
    dataFreezeAx.set_xlabel("x")
    dataFreezeAx.set_yticks([])  
    dataFreezeAx.yaxis.tick_right()
    axis.append(dataFreezeAx)
    
if SHOWNORMALIZED:
    plotNumber += 1
    dataNormalizedAx = fig.add_subplot(1,numberOfPlots,plotNumber)
    dataNormalizedContourf = dataNormalizedAx.contourf(x,t,dataArrayNormalized, levels=levels, cmap='coolwarm')
    
    dataNormalizedAx.set_title("normalized")    
    dataNormalizedAx.set_xlabel("x")
    dataNormalizedAx.set_yticks([])  
    dataNormalizedAx.yaxis.tick_right()
    axis.append(dataNormalizedAx)
    
colorbar = fig.colorbar(dataContourf,ax=axis, location='right')
    

ticksLinSpace = np.linspace(-LinftyDataArray,LinftyDataArray,NUMBEROFCOLORTICKS)
colorbar.set_ticks(ticksLinSpace)
dataAx.set_title(functionName)
dataAx.set_xlabel("x")
dataAx.set_ylabel("t")

fig.text(0.1, 0,"simulation script    " + pythonFileName + "\n"+"visualization script " +usedVisualizationScriptName, size=10, ha='left', va='bottom')

print(datetime.datetime.now(),"plot assigned") 





### show/export plot
print(datetime.datetime.now(),"plotting")
if DEBUGGING:
    plt.show()
else:
    plt.savefig(outputFolderPath+TITLE+"plot.png", dpi = 128, bbox_inches='tight')
print(datetime.datetime.now(),"exported")

### write info to file
if not(DEBUGGING):
    if os.path.isfile(originalInfoFilePath):
        copy(originalInfoFilePath, outputFolderPath+"info.txt")
infoFile = open(originalInfoFilePath,"a")
scriptEndTime = datetime.datetime.now()
scriptTime = scriptEndTime-scriptStartTime
print(scriptEndTime,"total visualization time"+" = \t"+str(scriptTime))
infoString += "\n\t"+"total visualization time"+" = \t"+str(scriptTime)
if not(DEBUGGING):
    infoFile.write("\n\n")
    infoFile.write(infoString)
    infoFile.close()
#plt.show()


print(datetime.datetime.now(),"finished")


