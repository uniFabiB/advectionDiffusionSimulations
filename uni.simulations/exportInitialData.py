import numpy as np
import os

output_dir_path = os.path.dirname(os.path.realpath(__file__))


xValues = 4096
scale = 1/1000

OutputArray = scale*2*(np.random.rand(xValues)-0.5)       # zwischen 0 und 1
np.save(output_dir_path +"/../data/initialData/temp", OutputArray)

print('done')