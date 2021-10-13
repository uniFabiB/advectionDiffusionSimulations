import numpy
from firedrake import *
import vtktools                 #vtktools.py von hier https://raw.githubusercontent.com/FluidityProject/fluidity/main/python/vtktools.py in firedrake/lib/python3.6/site-packages kopieren 
import os
import xml.etree.ElementTree as ElementTree
import meshio

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





mesh = meshio.read(dataFolderPath+timeFileArray[0][1])

V = FunctionSpace(mesh, "CG", 2)
X = interpolate(SpatialCoordinate(mesh), V)

vtu = vtktools.vtu("output_0.vtu")
reader = lambda X: vtu.ProbeData(numpy.c_[X, numpy.zeros(X.shape[0])], "Solution").reshape((-1,))

u = Function(V)
u.dat.data[:] = reader(X.dat.data_ro)

(x, y) = SpatialCoordinate(mesh)
w = interpolate(x + y, V)
print("||u - w||: ", norm(u - w))