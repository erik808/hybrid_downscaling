import numpy as np
import matplotlib
matplotlib.use('Qt5Cairo')
from matplotlib import pyplot as plt
from transectpicker import TransectPicker

fig, ax  = plt.subplots(figsize=(5,4))
A = np.random.rand(50,50)

im = plt.pcolormesh(A)

tpicker = TransectPicker(im, A)
plt.show()
