from PIL import Image
import numpy as np
import rasterio

# name of your image file
filename = 'imagery/lcm_split/train/input0_256.tif'

# open image using PIL
img = rasterio.open(filename)

# convert to numpy array
img = img.read()

# find number of channels
if img.ndim == 2:
    channels = 1
    print("image has 1 channel")
else:
    channels = img.shape[0]
    print("image has", channels, "channels")

"""
pix2pix_combine
Combine 2 images from different domains for pix2pix. Make sure images in folderA and folderB have the same name.
Folder Structure:
folderA
    |--> train
    |--> valid (if any)
    |--> test (if any)
folderB
    |--> train
    |--> valid (if any)
    |--> test (if any)
dest_path
    |--> train
    |--> valid (if any)
    |--> test (if any_
Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import os
import numpy as np
from PIL import Image
import rasterio
import cv2

# define paths for translation from domain A (images in folderA) -> domain B (images in folderB)
folderA = '/Users/adriaankeurhorst/Documents/MScThesis/imagery/lcm_split'
folderB = '/Users/adriaankeurhorst/Documents/MScThesis/imagery/sen2_split'
dest_path = '/Users/adriaankeurhorst/Documents/MScThesis/imagery/split_joined_lcm_sen2'
import matplotlib.pyplot as plt
splits = os.listdir(folderA)
for sp in splits:
    img_fold_A = os.path.join(folderA, sp)
    img_fold_B = os.path.join(folderB, sp)
    img_list = os.listdir(img_fold_A)
    num_imgs = len(img_list)
    img_fold_AB = os.path.join(dest_path, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A1 = rasterio.open(path_A)
            im_A = im_A1.read()
            im_A11 = np.reshape(im_A, (256, 256, 1))
            im_B1 = rasterio.open(path_B)
            im_B = im_B1.read()[:1]
            im_B11 = np.reshape(im_B, (256, 256, 1))
            # plt.figure()
            # plt.imshow(im_A11[:, :, 2])
            # plt.figure()
            # plt.imshow(im_B11[:,:,2])
            im_AB = np.concatenate([im_A11, im_B11], 0)
            # cv2.imwrite(path_AB, im_AB)
            try:
                # Create empty TIF image with dimensions of FIN but with name of FOUT.
                with rasterio.open(
                        path_AB,
                        'w',
                        driver='GTiff',
                        height=im_AB.shape[1],
                        width=im_AB.shape[2],
                        count=im_AB.shape[0],
                        dtype='float32',
                        crs=im_A1.crs,
                        transform=im_A1.transform,
                ) as dst:
                    dst.write(im_AB)
                    print(f"File created: {path_AB}")
            except IOError as e:
                print(f"Couldn't write a file at {path_AB}. Error: {e}")

# load, split and scale the maps dataset ready for training
from os import listdir
import keras.initializers.initializers_v1
import keras_preprocessing.image
from numpy import asarray
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from numpy import vstack
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from numpy import savez_compressed
# keras_preprocessing.image.img_to_array()
# load all images in a directory into memory
def normalize(arr):
	''' Function to scale an input array to [-1, 1]    '''
	arr = np.nan_to_num(arr)
	arr_min = arr.min()
	arr_max = arr.max()
    # Check the original min and max values
	print('Min: %.3f, Max: %.3f' % (arr_min, arr_max))
	arr_range = arr_max - arr_min
	scaled = np.array((arr-arr_min) / float(arr_range), dtype='float64')
	arr_new = -1 + (scaled * 2)
	# Make sure min value is -1 and max value is 1
	print('Min: %.3f, Max: %.3f' % (arr_new.min(), arr_new.max()))
	return arr_new

def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		# pixels = load_img(path + filename, target_size=size)
		pixels = rasterio.open(path + filename).read()
		# convert to numpy array
		# pixels = img_to_array(pixels)
		# split into satellite and map
		clim_img, sat_img = pixels[:256, :, :], pixels[256:, :, :]
		clim_img, sat_img = normalize(clim_img), normalize(sat_img)
		src_list.append(clim_img)
		tar_list.append(sat_img)
	return [asarray(src_list), asarray(tar_list)]

# dataset path
path = 'imagery/split_joined_lcm_sen2/test/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
## save as compressed numpy array
filename = 'lcm_sen2_test.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)