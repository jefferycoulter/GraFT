#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:11:55 2025

@author: isabella
"""

import numpy as np
import pandas as pd
import skimage
from skimage.filters import threshold_otsu
import scipy as sp
import matplotlib.pyplot as plt
import tifffile
import re
import sys
from skimage.exposure import rescale_intensity
import seaborn as sns
import scienceplots

plt.style.use(['science','nature']) # sans-serif font
plt.close('all')

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('axes', labelsize=12)
sizeL=12
params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
plt.rcParams.update(params)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False  # Disable LaTeX
mpl.rcParams['axes.unicode_minus'] = True  # Ensure minus sign is rendered correctly


cmap=sns.color_palette("colorblind")

###############################################################################
#
# Frangi vs Sato filter
#
###############################################################################
generate_new_image=0

path="/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/"
sys.path.append(path)

import dataUtils

plt.close('all')

box_size=500
num_lines = 10
curvature_factor = 50
noiseBlur=1
noiseBack = 0.01

###############################################################################
#
# create dataset with 5 lines
#
###############################################################################
plt.close('all')

if(generate_new_image==1):

    saveOverpath = '/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/frangi_sato/'
    
    
    
    df_1 = dataUtils.generate_filamentous_structure(box_size, num_lines, curvature_factor)  
    df_1['frame'] = 1
    
    
    
    
    savepath5 = saveOverpath + 'noise10/'
    files5 = ['100_lines_intermediate.csv','200_lines_intermediate.csv','300_lines_intermediate.csv','400_lines_intermediate.csv','500_lines_intermediate.csv']
    
    filename = re.sub(r'.csv', '', 'sato_vs_frangi.csv')
    
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_1,
                                                    noiseBlur=1,noiseBack=noiseBack,box_size=box_size,
                                                    path=savepath5,filename=filename)
    
    noise_img = dataUtils.generate_images_from_data(df_timeseries=df_1,
                                                    noiseBlur=1,noiseBack=noiseBack,box_size=box_size,
                                                    path=savepath5,filename=filename+'_nonoise',noise='no')
         



sigma = 2
thresh_top=0.5
small=5
with tifffile.TiffFile('/home/isabella/Documents/PLEN/dfs/insilico_datacreation/postdoc/frangi_sato/noise10/sato_vs_frangi.tiff') as tif:
    img_o = tif.asarray()
    
# we do it for image 1
image = img_o

plt.figure()
plt.imshow(image)

imI = 255.0 * (image - image.min()) / (image.max() - image.min())  
image1=imI 

imG=skimage.filters.gaussian(image1,sigma)   
# 2) frangi tubeness
imFrangi = skimage.filters.frangi(imG, sigmas=np.arange(1, 3, 0.1), scale_step=0.1, alpha=0.1, beta=2, gamma=15, black_ridges=False, mode='reflect', cval=0)

imSato = skimage.filters.sato(imG, sigmas=np.arange(1, 3, 0.1),black_ridges=False, mode='reflect', cval=0)

frangi_rescaled = rescale_intensity(imFrangi, out_range=(0, 1))
sato_rescaled = rescale_intensity(imSato, out_range=(0, 1))

otsu_thresh_frangi = threshold_otsu(frangi_rescaled)
otsu_thresh_sato = threshold_otsu(sato_rescaled)

binary_frangi = frangi_rescaled > otsu_thresh_frangi
binary_sato = sato_rescaled > otsu_thresh_sato

imFS = skimage.morphology.skeletonize(binary_frangi > 0)
imSS = skimage.morphology.skeletonize(binary_sato > 0)

plt.figure()
plt.imshow(imFS)

plt.figure()
plt.imshow(imSS)

plt.close('all')

plt.figure()
plt.imshow(image)

imE = skimage.exposure.equalize_adapthist(imFrangi, kernel_size=None, clip_limit=0.01, nbins=256)
# 5) median filter
imM = sp.ndimage.median_filter(imE, size=(2,2),  mode='reflect', cval=0.0, origin=0)
# 6) hysteresis
thresh = threshold_otsu(imM)
imH = skimage.filters.apply_hysteresis_threshold(imE, thresh*thresh_top, thresh)
# 7) morph grey closing again
#imC2 =skimage.morphology.grey.closing(imH*1, circle)
# 8) skeletonize
imS = skimage.morphology.skeletonize(imH > 0)
# 9) remove small objects
imageCleaned = skimage.morphology.remove_small_objects(imS, small, connectivity=2) > 0
plt.figure()
plt.imshow(imageCleaned)




imSRe = rescale_intensity(imSato, out_range=(0, 1))

plt.figure()
plt.imshow(imE)

imE2 = skimage.exposure.equalize_adapthist(imSRe, kernel_size=None, clip_limit=0.01, nbins=256)
# 5) median filter
imM2 = sp.ndimage.median_filter(imE2, size=(2,2),  mode='reflect', cval=0.0, origin=0)
# 6) hysteresis
thresh2 = threshold_otsu(imM2)
imH2 = skimage.filters.apply_hysteresis_threshold(imE2, thresh2*thresh_top, thresh2)
# 7) morph grey closing again
#imC2 =skimage.morphology.grey.closing(imH*1, circle)
# 8) skeletonize
imS2 = skimage.morphology.skeletonize(imH2 > 0)
# 9) remove small objects
imageCleaned2 = skimage.morphology.remove_small_objects(imS2, small, connectivity=2) > 0
plt.figure()
plt.imshow(imageCleaned2)


plt.close('all')

fig, axd = plt.subplot_mosaic("ABC", figsize=(8.27,3))
axd["A"].imshow(image,cmap='gray_r')
axd["B"].imshow(imageCleaned,cmap='gray_r')
axd["C"].imshow(imageCleaned2,cmap='gray_r')

axd["A"].axis("off")
axd["B"].axis("off")
axd["C"].axis("off")

for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.03, key, transform=ax.transAxes, 
            size=12, weight='bold')
 
keys_to_label = {
    'A': 'Original image',
    'B': 'Frangi filter',
    'C': 'Sato filter'
}
for key, label in keys_to_label.items():
    ax = axd[key]
    ax.text(0.5, 1.03, label, transform=ax.transAxes, 
            ha='center', size=12, weight='normal')
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/dfs/figures_article/figs/frangi-sato.pdf')
