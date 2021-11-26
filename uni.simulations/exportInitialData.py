import numpy as np
import os
import datetime

output_dir_path = os.path.dirname(os.path.realpath(__file__))


xValues = 1024
scale = 1
force0average = True

OutputArray = scale*2*(np.random.rand(xValues)-0.5)       # rand ergibt werte zwischen 0 und 1
if force0average:
    OutputArray -= np.average(OutputArray)

print("sum_i array[i]"+str(np.sum(OutputArray)))
scriptTimeStamp = str(datetime.datetime.now())
scriptTimeStamp = scriptTimeStamp.replace(":","-")
scriptTimeStamp = scriptTimeStamp.replace(" ","_")

print(OutputArray)
np.save(output_dir_path +"/initialData/"+scriptTimeStamp, OutputArray)

print('done')