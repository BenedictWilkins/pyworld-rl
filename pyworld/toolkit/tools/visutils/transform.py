import cv2
import numpy as np

'''
All transformations assume HWC image format (following the opencv convention).
'''

def scale(image, scale, interpolation=cv2.INTER_CUBIC):
    '''
        scales the given image
        Arguments:
            image: to scale
            scale: scale values (x,y)
            interpolation: default INTER_CUBIC
    '''
    if isinstance(scale, tuple):
        assert len(scale) == 2
        return cv2.resize(image, None, fx=scale[0], fy=scale[1], interpolation=interpolation)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

def resize(image, size, interpolation=cv2.INTER_CUBIC):
    if isinstance(size, tuple):
        assert len(size) == 2
        return cv2.resize(image, size, interpolation=interpolation)
    return cv2.resize(image, (size, size), interpolation=interpolation)

def crop(image, xsize, ysize, copy=True):
    image_c = image[ysize[0]:ysize[1], xsize[0]:xsize[1]]
    if copy:
        return np.copy(image_c)
    return image_c

def translate(image, x, y):
    M = np.array([[1,0,x],[0,1,y]], dtype=np.float32)
    return cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))

def rotate(image, theta, point=(0,0)):
    M = cv2.getRotationMatrix2D((point[1], point[0]), theta, 1)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def affine(image, p1, p2):
    assert p1.shape == p2.shape == (2,3)
    M = cv2.getAffineTransform(p1,p2)
    return cv2.warpAffine(image ,M, (image.shape[1], image.shape[0]))

def perspective(image, p1, p2):
    assert p1.shape == p2.shape == (2,4)
    M = cv2.getPerspectiveTransform(p1, p2)
    dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def colour(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def CHW(image): #TORCH FORMAT
    '''
        Converts an image (or collection of images) from HWC to CHW format.
    '''
    if len(image.shape) == 2:
        return image[np.newaxis,:,:]
    elif len(image.shape) == 3:    
        return image.transpose((2,0,1))
    elif len(image.shape) == 4:
        return image.transpose((0,3,1,2))
    else:
        raise ValueError("invalid dimension: " + str(len(image.shape)))
    
def HWC(image): #CV2 FORMAT
    '''
        Converts an image (or collection of images) from CHW to HWC format.
    '''
    if len(image.shape) == 2:
        return image[:,:,np.newaxis]
    if len(image.shape) == 3:    
        return image.transpose((1,2,0))
    elif len(image.shape) == 4:
        return image.transpose((0,2,3,1))
    else:
        raise ValueError("invalid dimension: " + str(len(image.shape)))

if __name__ == "__main__":
    image = np.random.uniform(size=(100,100,3))
    cv2.imshow('image', image)
 
    cv2.imshow('translate', translate(image, 10, 10))

    cv2.imshow('scale_in', scale(image, 0.1))

    cv2.imshow('scale_out', scale(image, 2))

    cv2.imshow('resize', resize(image, (20,20)))

    cv2.imshow('crop', crop(image, (20,80), (10,20)))

    while cv2.waitKey(60) != ord('q'):
        pass

