import numpy as np
import os
import datetime

output_dir_path = os.path.dirname(os.path.realpath(__file__))


xValues = 4096
scale = 1

OutputArray = scale*2*(np.random.rand(xValues)-0.5)       # rand ergibt werte zwischen 0 und 1

scriptTimeStamp = str(datetime.datetime.now())
scriptTimeStamp = scriptTimeStamp.replace(":","-")
scriptTimeStamp = scriptTimeStamp.replace(" ","_")

np.save(output_dir_path +"/../data/initialData/"+scriptTimeStamp, OutputArray)

print('done')