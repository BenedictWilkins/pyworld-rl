
from PIL import Image, ImageDraw, ImageFont

import numpy as np

import os

from . import transfrom

def character(char):
    '''
        Create an image of the given character.
        Arguments:
            char: to draw
    '''

    W = H = 14
    img = Image.new('RGB', (W,H), color = (0,0,0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(os.path.dirname(__file__) + "/res/ArialCE.ttf", 18)

    w,h = font.getsize(char)
    draw.text(((W-w)/2,(H-h)/2 - 2), char, font=font, fill=(255,255,255)) #dont ask...        
    return transform.CHW(transform.gray(np.array(img)) / 255.)