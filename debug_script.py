from h5py import File
import numpy as np

img_path = '/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/train/dr-arche-center-13/bl_000000008841.h5'
h5f = File(img_path, "r")

print(list(h5f.keys()))
data = np.array(h5f['dense_image_data'][0, :, :])
img = np.array(h5f['rgb_image_data'])
print(img.shape)

from matplotlib import pyplot as plt
plt.imshow(data, interpolation='nearest')
plt.savefig('test-depth-h5.png')

img = np.transpose(img, (1, 2, 0))

plt.imshow(img)
plt.savefig('test-img-bgr-h5.png')

print('Done')

import minexr
import imageio

img_path = '/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/ineveraray_loop30/depth/3941625000000.exr'
rgb_path = '/cluster/scratch/shimmi/aerial-depth-completion/data/datasets/ineveraray_loop30/color/3941625000000.png'

im = imageio.imread(rgb_path)
print('image ', im.shape)

# with open(img_path, 'rb') as fp:
#     reader = minexr.load(fp)
#     rgba = reader.select(['Color.R','Color.G','Color.B'])
#     # a HxWx3 np.array with dtype based on exr type.
#     print(rgb.shape())

import OpenEXR as exr # use "conda install -c conda-forge openexr-python"
import Imath

exrfile = exr.InputFile(img_path)
header = exrfile.header()
print('header is ', header)

dw = header["dataWindow"]
isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

channelData = dict()

# convert all channels in the image to numpy arrays
for c in header['channels']:
    C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
    C = np.fromstring(C, dtype=np.float32)
    C = np.reshape(C, isize)

    channelData[c] = C

print('channelData is ', channelData)

colorChannels = ['Z'] 
img = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)

# linear to standard RGB
img[..., :3] = np.where(img[..., :3] <= 0.0031308,
                        12.92 * img[..., :3],
                        1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)

print('img is ', img)
print('img shape is ', img.shape)

Z = None if 'Z' not in header['channels'] else channelData['Z']
print('Z is ', Z)
print('Z shape is ', Z.shape)

C = exrfile.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
C = np.fromstring(C, dtype=np.float32)
Z_oneline = np.reshape(C, isize)
print('Z_oneline is ', Z_oneline )

# sanitize image (so that max_depth = 0)
#img = np.where(img < 0.0, 0.0, np.where(Z.reshape(480,752,1) > 399, 0, img))

plt.imshow(img, interpolation='nearest')
plt.savefig('test-depth-exr.png')
print('Done 2 ')
