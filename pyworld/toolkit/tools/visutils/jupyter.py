import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
#pyo.init_notebook_mode()

from ipywidgets import Image, Layout, VBox, HBox, interact, IntSlider, IntProgress
import ipywidgets as widgets

from IPython.display import display, clear_output

import numpy as np

from . import transform
from . import plot as vis_plot

from .plot import line_mode

def progress(iterator, info=None):
    f = IntProgress(min=0, max=len(iterator), step=1, value=0) # instantiate the bar
    print(info)
    display(f)
    for i in iterator:
        yield i
        f.value += 1

def plot(x,y,mode=line_mode.line,legend=None):
    fig = vis_plot.plot(x,y, mode=mode, legend=legend)
    display(go.FigureWidget(fig))

def scatter_image(x, y, images, scale=1, scatter_colour=None, line_colour=None):
    #images must be in NHWC format
    assert transform.isHWC(images)

    if transform.is_float(images):
        images = transform.to_integer(images)

    fig = go.FigureWidget(data=[dict(type='scattergl',x=x,y=y,mode='markers+lines',
                marker=dict(color=scatter_colour),
                line=dict(color=line_colour))])
    fig.layout.hovermode = 'closest'
    scatter = fig.data[0]

    #convert images to png format
    image_width = '{0}px'.format(int(images.shape[2] * scale))
    image_height = '{0}px'.format(int(images.shape[1] * scale))
    images = [transform.to_bytes(image) for image in images]

    image_widget = Image(value=images[0], 
                        layout=Layout(height=image_height, width=image_width))
    def hover_fn(trace, points, state):
        ind = points.point_inds[0]
        image_widget.value = images[ind]

    scatter.on_hover(hover_fn)
    #fig.show()
    #print("WHAT")

    box_layout = widgets.Layout(display='flex',flex_flow='row',align_items='center',width='100%')
    display(HBox([fig, image_widget], layout=box_layout)) #basically... this needs to be done in jupyter..?!]
    return fig, image_widget

def images(images, scale=1):
    assert transform.isHWC(images)

    if transform.is_float(images):
        images = transform.to_integer(images)

    image_width = '{0}px'.format(images.shape[2] * scale)
    image_height = '{0}px'.format(images.shape[1] * scale)
    images = [transform.to_bytes(image) for image in images]
    image_widget = Image(value=images[0], layout=Layout(height=image_height, width=image_width))
    def slide(x):
        image_widget.value = images[x]
    interact(slide, x=IntSlider(min=0, max=len(images)-1, step=1, value=0))
    display(image_widget)

def image(image, scale=1):
    assert transform.isHWC(image)
    if transform.is_float(image):
        image = transform.to_integer(image)
    image_width = '{0}px'.format(image.shape[1] * scale)
    image_height = '{0}px'.format(image.shape[0] * scale)
    image_widget = Image(value=transform.to_bytes(image), layout=Layout(height=image_height, width=image_width))
    display(image_widget)

  