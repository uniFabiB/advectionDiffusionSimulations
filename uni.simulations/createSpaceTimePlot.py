import numpy
from firedrake import *
import vtk 
import os
import xml.etree.ElementTree as ElementTree
import meshio
from vtkmodules.vtkCommonCore import vtkDataArray
from networkx.generators.tests.test_small import null
from vtk.util.numpy_support import vtk_to_numpy
##### filestuff #####
scriptFilePath = os.path.dirname(os.path.realpath(__file__))
dataFolderPath = scriptFilePath+"/../data/loadData/"
pvdXMLTree = ElementTree.parse(dataFolderPath+"u.pvd")
pvdXMLTreeRoot = pvdXMLTree.getroot()

STARTTIME = 1
ENDTIME = 2

## create empty array
timeFileArrayTemp = [["",""] for y in range(len(pvdXMLTreeRoot[0]))]


i = 0
for dataset in pvdXMLTreeRoot[0]:
    timeStep = float(dataset.get('timestep'))
    if timeStep>=STARTTIME and timeStep<=ENDTIME:
        timeFileArrayTemp[i][0]= timeStep
        timeFileArrayTemp[i][1]= dataset.get('file')
        i+=1
numberOfFiles = i

timeFileArray = [["",""] for y in range(numberOfFiles)]
timeFileArray = timeFileArrayTemp[0:numberOfFiles][:]


print(dataFolderPath+timeFileArray[0][1])


data = meshio.read(dataFolderPath+timeFileArray[0][1])
L=max(data.points[:,0])
infoString = "infostring"
infoString += "\n\t"+"L"+" = \t\t"+str(L)
cells=len(data.cells[0])
points=len(data.points[:,0])

print("points =",points)
infoString += "\n\t"+"points"+" = \t\t"+str(points)
n=cells
print("n =",n)
infoString += "\n\t"+"n"+" = \t\t"+str(n)
degree = round(points/cells)
infoString += "\n\t"+"degree"+" = \t\t"+str(degree)

print(infoString)
#https://stackoverflow.com/a/54072929

# load a vtk file as input
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(dataFolderPath+timeFileArray[0][1])
reader.Update()

output = reader.GetOutput()


#entweder index (-> 0) oder den namen in <PointData Vectors="function_8[0]">     -> function_8[0]

potential = output.GetPointData().GetArray(0)
abc = output.GetPointData().GetArray("a")
# Get the coordinates of nodes in the mesh
nodes_vtk_array= reader.GetOutput().GetPoints().GetData()

test = vtk_to_numpy(potential)
print(test[:,0])

print(test)
import matplotlib.pyplot as plt
ones = np.ones(100)
print(potential.GetNumberOfTuples())
print(potential.GetTuple(0))


plt.figure()
plt.plot(test[:,0])
plt.show()


