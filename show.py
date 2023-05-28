import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import skimage.io as io
import numpy as np
from matplotlib import pylab as plt
import matplotlib


img=nib.load('23w3d.nii')

print(img.dataobj.shape)

width, height, queue = img.dataobj.shape
 
OrthoSlicer3D(img.dataobj).show()
 
x = int((queue/10) ** 0.5) + 1
num = 1

for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(x, x, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
 
plt.show()

