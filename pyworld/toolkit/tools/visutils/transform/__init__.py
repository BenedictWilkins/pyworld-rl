import cv2
import numpy as np
import skimage.transform

from types import SimpleNamespace

from .shape import nform, isHWC, isCHW, image_format


'''
All transformations assume HWC float32 image format (following the opencv convention).
'''

interpolation = SimpleNamespace(nearest=0, bilinear=1, biquadratic=2, bicubic=3, biquartic=4, biquintic=5)


def resize(image, width, height, interpolation=interpolation.nearest):
    """
        Resize image(s).
    """
    image, r = nform(image)
    size = list(image.shape)
    size[1], size[2] = height, width
    return r(skimage.transform.resize(image, size, order=interpolation))

def scale(image, scale, *scaleh, interpolation=interpolation.nearest): #width, height
    """ Scale image(s)

    Args:
        image (numpy.ndarray): image(s) to scale
        scale (int, float): scale of the image
        scaleh (int, float, optional) height scale of the image, if given scale will be used as the width scale.
    Returns:
        np.ndarray: scaled image(s)
    """
    assert len(scaleh) <= 1
    scale = (scale, *scaleh)
    if len(scale) == 1:
        scale = (scale[0], scale[0])

    image, r = nform(image)
    size = list(image.shape) #NHWC
    size[1], size[2] = int(size[1] * scale[1]), int(size[2] * scale[0])
    return r(skimage.transform.resize(image, size, order=interpolation))


def crop(image, xsize=None, ysize=None, copy=True):
    """ Crop image(s).

    Args:
        image ([numpy.ndarray]): image to crop in (N)HW(C) format
        xsize ([tuple], optional): crop width (lower, upper) index. Defaults to None.
        ysize ([tuple], optional): crop height (lower, upper) index. Defaults to None.
        copy (bool, optional): Create a new array or not. Defaults to True.

    Returns:
        numpy.ndarray: cropped image(s)
    """
    image, r = nform(image)

    if xsize is None:
        xsize = (0, image.shape[2])
    if ysize is None:
        ysize = (0, image.shape[1])

    assert isinstance(xsize, tuple) and len(xsize) == 2
    assert isinstance(ysize, tuple) and len(ysize) == 2

    image = image[:,ysize[0]:ysize[1], xsize[0]:xsize[1],:]
    if copy:
        image = np.copy(image)
    return r(image)

def grey(image, components=(0.299, 0.587, 0.114)): #(N)HW(C) format
    """ Image(s) to grayscale

    Args:
        image (ndarray): colour image to be converted to grayscale
        components (tuple, optional): scale components for colour channels. Defaults to (0.299, 0.587, 0.114).

    Returns:
        ndarray: gray scaled image(s)
    """
    image, r = nform(image)
    assert image.shape[-1] == 3 # image must colour format

    return r((image[...,0] * components[0] + image[...,1] * components[1] + image[...,2] * components[2])[...,np.newaxis].astype(image.dtype))

def colour(image, components=(1,1,1)):
    """
    Args:
        image ([type]): [description]
        components (tuple, optional): [description]. Defaults to (0.299, 0.587, 0.114).

    Returns:
        [type]: [description]
    """
    image, r = nform(image)
    return r(image * np.array(components)[np.newaxis, np.newaxis, np.newaxis, :])


### BELOW ONLY WORK FOR A SINGLE IMAGE - TODO

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


     


def binary(image, threshold=0.5, **kwargs):
    i = (int(is_integer(image)))
    m = (1.,255)[i]
    t = (threshold, 255 * threshold)[i]
    indx = image > t
    image[indx] = m
    image[np.logical_not(indx)] = 0
    return image

def to_bytes(image, ext='.png'):
    if ext is not None:
        success, image = cv2.imencode(ext, image)
        if success:
            return image.tobytes()
        else:
            raise ValueError("failed to convert image to bytes in format: {0}".format(ext))
    else:
        return image.tobytes() #just use numpy...?

#---------------- BATCH_TRANSFORMATIONS

#TODO others?

def binary_all(images, threshold=0.5):
    return binary(images, threshold=threshold)

def gray_all(image, components=(0.299, 0.587, 0.114)): #(N)HWC format
    return gray(image, components=components)

def crop_all(images, xsize=None, ysize=None, copy=False):
    if not len(images.shape) == 4 or not isHWC(images):
         raise ValueError("invalid image format: {0}, images must be in NHWC format.".format(images.shape))
    if xsize is None:
        xsize = (0,images.shape[2])
    if ysize is None:
        ysize = (0,images.shape[1])

    assert isinstance(xsize, tuple) and len(xsize) == 2
    assert isinstance(ysize, tuple) and len(ysize) == 2

    image_c = images[:,ysize[0]:ysize[1], xsize[0]:xsize[1]]
    if copy:
        return np.copy(image_c)
    return image_c





def __format_to_index__(format): # default format is HWC
    (format.index("H"), format.index("W"), format.index("C"))




    
def CHW(image): #TORCH FORMAT
    '''
        Converts an image (or collection of images) from HWC to CHW format.
        CHW format is the image format used by PyTorch.
    '''
    if len(image.shape) == 2: #assume HW format
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
        HWC format is the image format used by PIL and opencv.
    '''
    if len(image.shape) == 2:
        return image[:,:,np.newaxis]
    if len(image.shape) == 3:    
        return image.transpose((1,2,0))
    elif len(image.shape) == 4:
        return image.transpose((0,2,3,1))
    else:
        raise ValueError("invalid dimension: " + str(len(image.shape)))

def BGR(image):
    raise NotImplementedError("TODO")
    #return np.flip(image, 2) #flip around channel axis

def RGB(image):
    raise NotImplementedError("TODO") #flip around channel axis

def is_integer(image):
    return issubclass(image.dtype.type, np.integer)

def is_float(image):
    return issubclass(image.dtype.type, np.floating)

def to_float(image):
    if is_float(image):
        return image.astype(np.float32)
    elif is_integer(image):
        return image.astype(np.float32) / 255.
    else:
        return TypeError("Invalid array type: {0} for float32 conversion.".format(image.dtype))

def to_integer(image):
    if is_integer(image):
        return image.astype(np.uint8) #check overflow?
    elif is_float(image):
        return (image * 255.).astype(np.uint8) 
    else:
        return TypeError("Invalid array type: {0} for uint8 conversion.".format(image.dtype))

if __name__ == "__main__":
    def test_isHWC():
        a = np.random.randint(0,255,size=(10,10))
        assert not isHWC(a)
        a = np.random.randint(0,255,size=(10,10,1))
        assert isHWC(a)
        a = np.random.randint(0,255,size=(100,10,10,1))
        assert isHWC(a)
    test_isHWC()



    '''
    image = np.random.uniform(size=(100,100,3))

    cv2.imshow('image', image)
 
    cv2.imshow('translate', translate(image, 10, 10))

    cv2.imshow('scale_in', scale(image, 0.1))

    cv2.imshow('scale_out', scale(image, 2))

    cv2.imshow('resize', resize(image, (20,20)))

    cv2.imshow('crop', crop(image, (20,80), (10,20)))

    cv2.imshow('gray', gray(image))

    cv2.imshow('binary', binary(gray(image)))

    while cv2.waitKey(60) != ord('q'):
        pass
    '''



"""
    
def scale_all(images, _scale, interpolation=interpolation.area, inplace=False):
    if not len(images.shape) == 4 or not isHWC(images) or is_integer(images):
        raise ValueError("invalid image format: {0} {1}, images must be in float32 NHWC format.".format(images.dtype, images.shape))
    
    if isinstance(_scale, tuple):
        assert len(_scale) == 2
    elif isinstance(_scale, (float, int)):
        _scale = (_scale,_scale)
    else:
        raise ValueError("invalid argument: scale {0}".format(_scale))

    nw = int(images.shape[2] * _scale[0])
    nh = int(images.shape[1] * _scale[1])

    def scale_all_inplace():
        for i in range(images.shape[0]):
            images[i,:nh,:nw,:] = scale(images[i], _scale,  interpolation=interpolation)
        return images[:,:nh,:nw,:]

    def scale_all():
        result = np.empty((images.shape[0], nh, nw, images.shape[3]), dtype=np.float32)
        for i in range(images.shape[0]):
            result[i] = scale(images[i], _scale, interpolation=interpolation)
        return result

    return (scale_all, scale_all_inplace)[int(inplace)]()

def resize_all(images, size, interpolation=interpolation.area):
    if not len(images.shape) == 4 or not isHWC(images) or is_integer(images):
        raise ValueError("invalid image format: {0} {1}, images must be in float32 NHWC format.".format(images.dtype, images.shape))
    if isinstance(size, tuple):
        assert len(size) == 2
    elif isinstance(size, int):
        size = (size,size)
    else:
        raise ValueError("invalid argument: size {0}".format(size))
 
    def resize_all():
        result = np.empty((images.shape[0], size[1], size[0], images.shape[3]), dtype=np.float32)
        for i in range(images.shape[0]):
            result[i] = cv2.resize(images[i], size, interpolation=interpolation).reshape(size[1], size[0], images.shape[3])
        return result

    return resize_all()

 
def scale(image, scale, interpolation=interpolation.nearest):
    '''
        scales the given image
        Arguments:
            image: to scale
            scale: scale values (x,y)
            interpolation: default INTER_NEAREST
    '''
    if isinstance(scale, int,float):
        scale = (scale, scale)
    result = cv2.resize(image, None, fx=scale[0], fy=scale[1], interpolation=interpolation)
    result = result.reshape(result.shape[0], result.shape[1], scale.shape[2])



def resize(image, size, interpolation=interpolation.area):
    if isinstance(size, tuple):
        assert len(size) == 2
        return cv2.resize(image, size, interpolation=interpolation)
    return cv2.resize(image, (size, size), interpolation=interpolation)
"""
