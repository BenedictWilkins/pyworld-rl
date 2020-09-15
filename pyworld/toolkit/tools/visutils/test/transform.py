from pyworld.toolkit.tools.visutils import transform as T
import numpy as np


nhwc3i = np.random.randint(0,255,(10,20,20,3))
nhwc1i = np.random.randint(0,255,(10,20,20,1))
nhwi = np.random.randint(0,255,(10,20,20))
hwc3i = np.random.randint(0,255,(20,20,3))
hwc1i = np.random.randint(0,255,(20,20,1))
hwi = np.random.randint(0,255,(20,20))

print("RESIZE (5,10)")
print(nhwc3i.shape, T.resize(nhwc3i, 5,10).shape)
print(nhwc1i.shape, T.resize(nhwc1i, 5,10).shape)
print(nhwi.shape, T.resize(nhwi, 5,10).shape)

print(hwc3i.shape, T.resize(hwc3i, 5, 10).shape)
print(hwc1i.shape, T.resize(hwc1i, 5, 10).shape)
print(hwi.shape, T.resize(hwi, 5, 10).shape)


print("SCALE (.5,.2)")
print(nhwc3i.shape, T.scale(nhwc3i, .5, .2).shape)
print(nhwc1i.shape, T.scale(nhwc1i, .5, .2).shape)
print(nhwi.shape,   T.scale(nhwi,   .5, .2).shape)

print(hwc3i.shape, T.scale(hwc3i, .5, .2).shape)
print(hwc1i.shape, T.scale(hwc1i, .5, .2).shape)
print(hwi.shape,   T.scale(hwi,   .5, .2).shape)


print("CROP (8,12), (5,7)")
print(nhwc3i.shape, T.crop(nhwc3i, (8,12), (5,7)).shape)
print(nhwc1i.shape, T.crop(nhwc1i, (8,12), (5,7)).shape)
print(nhwi.shape,   T.crop(nhwi,   (8,12), (5,7)).shape)

print(hwc3i.shape, T.crop(hwc3i, (8,12), (5,7)).shape)
print(hwc1i.shape, T.crop(hwc1i, (8,12), (5,7)).shape)
print(hwi.shape,   T.crop(hwi,   (8,12), (5,7)).shape)