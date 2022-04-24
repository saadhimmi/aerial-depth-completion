import numpy as np
from numpy.lib.type_check import mintypecode
import OpenEXR as exr
import matplotlib.pyplot as plt
import Imath
import imageio


img_path = "/home/shimmi/Desktop/1640723131024235963.exr"
exrfile = exr.InputFile(img_path)
# exrfile = exr.InputFile("/media/shimmi/Elements/irchel-bags/final/h0p90/depth/1639925328856065988.exr")

dw = exrfile.header()["dataWindow"]
isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
depth = exrfile.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
depth = np.fromstring(depth, dtype=np.float32)
depth = np.reshape(depth, isize)

print(depth.shape)

midx = 2
midy = 2
minx = (midx - 2)
maxx = (midx + 2)
miny = (midy - 2)
maxy = (midy + 2)

print('x from ', minx ,' to ', maxx)
print('y from ', miny,' to ', maxy)

print(depth[minx:maxx,miny:maxy])

