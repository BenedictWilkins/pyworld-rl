import cv2
import numpy as np

'''
All transformations assume HWC float32 image format (following the opencv convention).
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

def gray(image, components=(0.299, 0.587, 0.114)): #(N)HWC format
    return (image[...,0] * components[0] + image[...,1] * components[1] + image[...,2] * components[2])[...,np.newaxis]

    #return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ?? hmm..

def colour(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def binary(image, threshold=0.5):
    indx = image > threshold
    image[indx] = 1.
    image[np.logical_not(indx)] = 0.
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

def scale_all(images, _scale, interpolation=cv2.INTER_CUBIC, inplace=False):
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

def __is_channels__(axes):
    return axes == 1 or axes == 3 or axes == 4

def isCHW(image):
    '''
        Is the given image in HWC or NHWC.
        Arguments:
            image: to check
    '''
    C_index = 4 - len(image.shape)
    if C_index in [0,1] and __is_channels__(image.shape[C_index]):
        return True
    return False

def isHWC(image):
    '''
        Is the given image in HWC or NHWC.
        Arguments:
            image: to check
    '''
    C_index = 4 - len(image.shape)
    if C_index in [0,1] and __is_channels__(image.shape[-1]):
        return True
    return False
    
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

def is_integer(image):
    return issubclass(image.dtype.type, np.integer)

def is_float(image):
    return issubclass(image.dtype.type, np.floating)

def to_float(image):
    assert is_integer(image)
    return image.astype(np.float32) / 255.

def to_integer(image):
    assert is_float(image)
    return (image * 255.).astype(np.uint8) 

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
