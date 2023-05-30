# cGAN-for-snow
Before starting, ensure you have an idea of what you want the input and target datasets to be. For my thesis, the input was a 16 band image containing environmental data and the target was a Landsat-8 image.

# 1. resample.py
First, resample the input and target images to match the resolution of the finest one. If the target image, in this study a Landsat-8 image, has the finest resolution, resample the input image to match its resolution with this script.

# 2. split.py
Split the images into 256x256 pixel sized images with this script. If everything goes well and the input images were the same dimensions, you should get the same number of output images for both the input and target images. Check to make sure.

# 3. tiff_to_arr.py
This script converts the split 256x256 pixel input and target images into a compressed numpy array. Ensure you have a training and testing folder for both the input and target imagery, with the split images in them. It takes the directory of the input and target imagery as arguments, matches the images by split index, and then gives you a single output npz. Do this separately for the training and testing datasets.

# 4. willie.py
Don't mind the name please, but this script is the actual cGAN image generation. It takes the npzs you have made in the previous script as input, and then trains a specified number of steps (reccomend at least 100k steps), and then produces a final output image. It also saves every 5k training steps.
