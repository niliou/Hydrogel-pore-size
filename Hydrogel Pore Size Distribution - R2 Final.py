##Authors: Maryam Jamshidi (nasstaran.jamshidi@gmail.com) , Cavus Falamaki (c.falamaki@aut.ac.ir)
###Authors Acknowledge Niloufar Jamshidi for her helps (niloufar.jamshidy@gmail.com) Nov. 2020
import os
import numpy as np
import cv2 as cv
import skimage
import glob
import csv
import scipy as sp
import matplotlib 
from matplotlib import pyplot as plt
from scipy import ndimage, misc
from skimage import io, color, measure
from skimage.measure import label, regionprops, regionprops_table
from skimage.data import gravel
from skimage import filters
from skimage.segmentation import clear_border
from skimage.filters import difference_of_gaussians, meijering, sato, frangi, hessian, window
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.fftpack import fftn, fftshift
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 1: reading and loading the image from directory ##############################################################

PATH = 'D:/1-Labview image/python images/R7/C40i.tif'
name = os.path.split(PATH)[-1]
PATH = 'D:/1-Labview image/python images/R7/'  #### specify the path to save the images
pixel_to_um = 1                                     #### specify scale of the image
image = cv.imread('D:/1-Labview image/python images/R7/C40i.tif',0)
(W , H) = image.shape
# iimage = 255 - image    ### comment/uncomment to invert the look up table
# image = iimage          ### comment/uncomment to invert the look up table

# 2: Normalize the image #######################################################################################

mean, STD  = cv.meanStdDev(image)
offset = 1
clipped = np.clip(image, mean - offset*STD, mean + offset*STD).astype(np.uint8)
Nimage = cv.normalize(clipped,  None, 0, 255, cv.NORM_MINMAX)
cv.imwrite(os.path.join(PATH+'1-Normalized_'+name), Nimage)

# 3: Gaussian Blure and Edge detection filtering ###############################################################

kwargs = {'sigmas': [0.7], 'mode': 'reflect'}
Gimage = filters.gaussian(Nimage, sigma=0.7)
cv.imwrite(os.path.join(PATH+'2-Gaussian  filter_'+name), Gimage)

GN = cv.normalize(Gimage,  None, 0, 255, cv.NORM_MINMAX)
GNN = GN.astype(np.uint8)                                      ### APPLYING Gaussian smooth filter

Eimage = filters.sobel(GN)
cv.imwrite(os.path.join(PATH+'3- Sobel Edge detection_'+name), Eimage)

Eimage += Nimage
EN = cv.normalize(Eimage,  None, 0, 255, cv.NORM_MINMAX)
ENN = EN.astype(np.uint8)                                       ### APPLYING Sobel edge detection filter

DoG = filters.difference_of_gaussians(Gimage,0.7)
# cv.imwrite(os.path.join(PATH+'DoG_'+name), DoG)
DoG += Nimage
DoGN = cv.normalize(DoG,  None, 0, 255, cv.NORM_MINMAX)
DoGNN = DoGN.astype(np.uint8)                                   ### APPLYING Difference of Gaussian filter


LoG = ndimage.gaussian_laplace(Gimage, sigma=0.7)
# cv.imwrite(os.path.join(PATH+'LoG_'+name), LoG)
LoG += Nimage
LoGN = cv.normalize(LoG,  None, 0, 255, cv.NORM_MINMAX)
LoGNN = LoGN.astype(np.uint8)                                   ### APPLYING Laplacian of Gaussian filter


GGM = ndimage.gaussian_gradient_magnitude(Gimage, sigma=0.7)
# cv.imwrite(os.path.join(PATH+'GGM_'+name), GGM)
GGM += Nimage
GGMN = cv.normalize(GGM,  None, 0, 255, cv.NORM_MINMAX)          
GGMNN = GGMN.astype(np.uint8)                                   ### APPLYING Gaussian Gradient Magnitude filter

eigen = np.linalg.eigvals(Gimage[0:image.ndim, 0:image.ndim])
HoEig= filters.hessian(Gimage, **kwargs)
# cv.imwrite(os.path.join(PATH+'HoEig_'+name), HoEig)
HoEig += Nimage
HoEigN = cv.normalize(HoEig,  None, 0, 255, cv.NORM_MINMAX)
HoEigNN = HoEigN.astype(np.uint8)                                ### APPLYING Hessian of gaussian eigenvalues filter

##########################################################################FFT Bandpass Filter################

rows, cols = GNN.shape
crow, ccol = int(rows / 2), int(cols / 2) 
# apply mask and inverse DFT
f = np.fft.fft2(GNN)                  #do the fourier transform
fshift1 = np.fft.fftshift(f)             #shift the zero to the center
maskNew = np.zeros((rows, cols), np.uint8)
maskNew[(np.abs(fshift1) > 10**3)&(np.abs(fshift1) < 10**20)] = 1  
fshift = fshift1 * maskNew                #Apply the mask
fftmask = np.abs(fshift)
fshift_mask_mag = np.log(fshift)
f_ishift = np.fft.ifftshift(fshift)      #inverse shift
img_back = np.fft.ifft2(f_ishift)        #inverse fourier transform
img_back2 = np.abs(img_back)
fftimage = np.abs(fshift1)

BPFN = cv.normalize(img_back2,  None, 0, 255, cv.NORM_MINMAX)     ## APPLYING BAND PASS FILTER
BPFNN = BPFN.astype(np.uint8)


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

Original = ax[0,0].imshow(image, cmap = plt.cm.gray)
ax[0,0].set_title('Original Image')

FFT = ax[0,1].imshow(np.log(fftimage), cmap=plt.cm.jet)
ax[0,1].set_title('Original FFT Magnitude (log)')

FFTmask = ax[1,1].imshow(np.log(fftmask), cmap=plt.cm.jet)
ax[1,1].set_title('FFT Mask Magnitude (log)')

Bandpass = ax[1,0].imshow(img_back2, cmap=plt.cm.gray)
ax[1,0].set_title('Bandpass Filter Result')


fig.colorbar(Original, ax=ax[0,0])
fig.colorbar(FFT, ax=ax[0,1])
fig.colorbar(FFTmask, ax=ax[1,1])
fig.colorbar(Bandpass, ax=ax[1,0])
cv.imwrite(os.path.join(PATH+'4- BandPassFilter_'+name), img_back2)


# 4: Threshold, morphological correction and Label particles  ##################################################

# retG,thG = cv.threshold(ENN , 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# thG = cv.adaptiveThreshold(DoGNN, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 15) #### for pore detection
thG = cv.adaptiveThreshold(DoGNN, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, 3) #### for hydrogel walls


se1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
se2 = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
se3 = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
kernel = np.ones((3,3), np.uint8)


Morphimage = cv.erode(thG , se2, iterations = 1)
for number in range(1):
    Morphimage = cv.dilate(Morphimage , se1, iterations = 1)
    Morphimage = cv.erode(Morphimage , se2, iterations = 1)

clearimage = clear_border(Morphimage)        ### comment the line if hydrogel wall thickness is measured

# THmask = Morphimage == 255
# labeledmask, numlabels = ndimage.label(THmask, structure = kernel)
# colorlabelmask = color.label2rgb(labeledmask, bg_label = 0)


# cv.imshow('Threshold', thG)
# cv.imshow('Threshold Adaptive', thG)
cv.imshow('Morph', Morphimage)
# cv.imwrite(os.path.join(PATH+'5- Threshold_'+name), thG)
# cv.imwrite(os.path.join(PATH+'7- Cleared_'+name), clearimage)
# cv.imwrite(os.path.join(PATH+'6- Morphological correction_'+name), Morphimage)

# 4: Aplying watershed to separate pores #######################################################################

sure_bg = cv.dilate(clearimage , se3, iterations = 2)
DT = cv.distanceTransform (clearimage , cv.DIST_L2, 3)
DT = DT.astype(np.uint8)
ret2, sure_fg = cv.threshold (DT, 0.025* DT.max(), 255, 0) 
unknown = cv.subtract(sure_bg, sure_fg)
ret3, markers = cv.connectedComponents (sure_fg)
markers = markers + 10
markers [unknown == 255] = 0
ENN = cv.cvtColor(DoGNN, cv.COLOR_GRAY2BGR)
markers = cv.watershed (ENN, markers)
ENN [markers == -1] = [0, 255, 255]
seg = color.label2rgb(markers, bg_label=0)

cv.imshow('overlay', ENN)
cv.imshow('Segmented pores', seg)
# cv.imwrite(os.path.join(PATH+'8- overlay_'+name), ENN)
# cv.imwrite(os.path.join(PATH+'9- Segmented pores_'+name), seg)

# 4: Saving pores measured features in csv file ################################################################

pores = measure.regionprops(markers , intensity_image = DoGNN)

proplist = ['Area', 'Centroid X','Centroid Y', 'equivalent_diameter', 'Orientation', 
            'MajorAxisLength', 'MinorAxisLength','Perimeter',
           'MinIntensity', 'MeanIntensity', 'MaxIntensity']

OutputResult = open(os.path.join(PATH + name + 'ImagePoresResult.csv'), 'w', newline='')
OutputResult.write((',' + ",".join(proplist) + '\n'))

void = 0

for pores_prop in pores:
    
    OutputResult.write(str(pores_prop['Label']))     
    for i,prop in enumerate(proplist):
        if (prop == 'Area') & (pores_prop.area * pixel_to_um **2 < 1000):
                to_print = pores_prop.area * pixel_to_um **2       #### Area pixel to um2
                Area = pores_prop.area * pixel_to_um **2
                void = void + Area  
        elif (prop == 'Centroid X'):
                to_print = pores_prop.centroid[0]
        elif (prop == 'Centroid Y'):
                to_print = pores_prop.centroid[1]
        elif (prop == 'Orientation'):
                to_print = pores_prop.orientation * 57.2958        ##### Radians to Degrees
        elif (prop.find('Intensity') <0):
                to_print = pores_prop[prop] * pixel_to_um          ##### without intensity in its name
        else:
            to_print = pores_prop[prop]                  
                     
        OutputResult.write(',' + str(to_print))               
    OutputResult.write('\n')
    
cv.waitKey(0)


# 5: Calculating void fraction (pores area/ total area) ########################################################

print('void fraction % = ' , void/(W*H*pixel_to_um **2)*100)




