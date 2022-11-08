from os import listdir
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import rasterio
import click
from numpy import savez_compressed
# load all images in a directory into memory
# dataset path
# pathA = 'imagery/2014_climate_data/WSS1/train/'
# pathB = 'imagery/L8/WSS1/train/'
# outpath = 'npzs/clim_l8_4SZN_train1.npz'
#
# def normalize(arr):
#     ''' Function to scale an input array to [-1, 1]    '''
#     arr = np.nan_to_num(arr)
#     arr_min = arr.min()
#     arr_max = arr.max()
#     # Check the original min and max values
#     print('OLD Min: %.3f, Max: %.3f' % (arr_min, arr_max))
#     arr_range = arr_max - arr_min
#     scaled = np.array((arr-arr_min) / float(arr_range), dtype='float64')
#     arr_new = -1 + (scaled * 2)
#     # Make sure min value is -1 and max value is 1
#     print('NEW Min: %.3f, Max: %.3f' % (arr_new.min(), arr_new.max()))
#     return arr_new

def load_images(patha, pathb, outpath):
    src_list, tar_list = list(), list()
  # enumerate filenames in directory, assume all are images
    for filename in listdir(patha):
      # load and resize the image
      # pixels = load_img(path + filename, target_size=size)
      pixelsA = rasterio.open(patha + filename).read()
      pixelsB = rasterio.open(pathb + filename).read()
      # plt.imshow(pixelsB[0])
      pixelsB1 = pixelsB.transpose((-1, -2, -3))
      # red = pixelsB1[2].flatten()
      # green = pixelsB1[1].flatten()
      # blue = pixelsB1[0].flatten()
      # rgb = np.concatenate((red, green, blue))
      # rgb1 = np.reshape(rgb, (256, 256, 3))
      # plt.imshow(pixelsB1)
      # convert to numpy array
      # pixels = img_to_array(pixels)
      # split into satellite and map
      clim_img, sat_img = pixelsA, pixelsB
      # clim_img, sat_img = normalize(clim_img), normalize(sat_img)
      src_list.append(clim_img)
      tar_list.append(sat_img)
    print('Loading da ting')
    src_list, tar_list = np.array(src_list), np.array(tar_list)
    src_images, tar_images = np.nan_to_num(src_list), np.nan_to_num(tar_list)
    print('Loaded: ', src_images.shape, tar_images.shape)
    # save as compressed numpy array
    filename = outpath
    savez_compressed(filename, src_images, tar_images)
    print('Saved dataset: ', filename)

    # return [asarray(src_list), asarray(tar_list)]
#
# [src_images, tar_images] = load_images(pathA, pathB)
# src_images, tar_images = np.nan_to_num(src_images), np.nan_to_num(tar_images)
# print('Loaded: ', src_images.shape, tar_images.shape)
# # save as compressed numpy array
# filename = outpath
# savez_compressed(filename, src_images, tar_images)
# print('Saved dataset: ', filename)

#
@click.command()
@click.argument('patha', type=click.Path(exists=True))
@click.argument('pathb', type=click.Path(exists=True))
@click.argument('outpath', type=click.Path())

def init(patha, pathb, outpath):
    '''
    :param pathA: the test or train folder of the independent variable(s)
    :param pathB: the test or train folder of the dependent variable(s)
    :param outpath: name that the array file should have + dir like npzs/lcm_clim_sen2_train.npz
    :return:
    '''
    load_images(patha, pathb, outpath)

if __name__ == "__main__":
    init()


