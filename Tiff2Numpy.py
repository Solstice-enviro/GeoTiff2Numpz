# Script by Alex Onojeghuo (PhD)
# Purpose - GeoTiff to numpy file convertion (batch conversion)
# Date - Feb. 28, 2021

# Load packages to enable numpy file creation and visualization
import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt
import os
import numpy as np
import glob
from numpy import savez_compressed

# Specify folder path for data
path = 'D:/MILA_UNET_DATA/Tiff/*.tif'
out = 'D:/MILA_UNET_DATA/Output' # specify output folder

# Create a loop that opens GeoTiff and saves as numpy zipped files
for index, filename in enumerate(glob.glob(path)):
    # Get input Raster properties
    a = rasterio.open(filename)
    arr = a.read()
    arr.shape
    basename = os.path.splitext(os.path.basename(filename))[0]
    #Specify path to output folder to save numpy zipped files
    os.chdir(out)
    # Save numpy files
    savez_compressed(f'{basename}.npz',arr)

# Preview npz shape
import os
import numpy as np
out = 'D:/MILA_UNET_DATA/Output'
os.chdir(out)
# Multiband dataset - feature
region1 = "region1.npz"
images = np.load(region1)[arr_0]
images.shape

# Label layer
region1_label = "region1_label.npz"
label = np.load(region1_label)["arr_0"]
label.shape
