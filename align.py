import numpy as np
from scipy import misc

y0 = misc.imread('data/mask/m-002-8.jpg')
x_recon0 = misc.imread('data/mask/m-002-1.jpg')

y0 = y0[99:500,249:550,:]
x_recon0 = x_recon0[99:500,249:550,:]
misc.imsave('data/mask/m1_1.jpg',y0)
misc.imsave('data/mask/m1_2.jpg',x_recon0)
