import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Set notebook mode to work in offline
#pyo.init_notebook_mode()

from ipywidgets import Image, Layout, VBox, HBox, interact, IntSlider, IntProgress, HTML, Output
import ipywidgets as widgets

from IPython.display import display, clear_output

import numpy as np
import math

from . import transform
from . import plot as vis_plot

from .plot import line_mode

try:
    from ipyevents import Event 
except:
    pass

try:
    from ipycanvas import Canvas, MultiCanvas
except:
    pass



class SimplePlot(vis_plot.SimplePlot):
    '''
        Wrapper around a plotly figure that allows offline updates in Jupyter.
    '''

    def __init__(self, x, y=None, *args, mode=line_mode.line, legend=None, **kwargs):
        if y is None:
            y = x 
            x = np.arange(y.shape[0])

        super(SimplePlot, self).__init__(x, y, *args, mode=mode, legend=legend, **kwargs)
        self.fig = go.FigureWidget(self.fig)
    
    def display(self):
        display(self.fig)
        return self

    def on_hover(self, fun, trace=0):
        self.fig.layout.hovermode = 'closest'
        self.fig.data[trace].on_hover(fun)

class DynamicPlot(SimplePlot):

    def __init__(self, x=[], y=[], update_after=100, *args, **kwargs):
        """ Create a dynamic plot that can be updated with new data.

        Args:
            x (list, optional): x data. Defaults to [].
            y (list, optional): y data. Defaults to [].
            update_after (int, optional): number of updates before replot. Defaults to 100.
        """
        super(DynamicPlot, self).__init__(x, y, *args,**kwargs)
        self.__cachex = []
        self.__cachey = [] 
        self.__update_after = update_after
        self.__count = 0

    def length(self, trace=0):
        return super(DynamicPlot, self).length(trace=trace) + len(self.__cachex)

    def extend(self, x=None, y=None, trace=0):
        """ Update this plot with new data (appends to the end of the trace). If x is left as None, the trace size is used.

        Args:
            x (iterable, optional): x data. Defaults to current trace size.
            y (iterable): y data.
            trace (int, optional): to update. Defaults to 0.

        """
        if y is None:
            y = x
            x = list(np.arange(self.__count, self.__count + len(y)))

        assert len(x) == len(y)
        assert trace == 0 # multiple traces not supported yet

        self.__cachex.extend(x)
        self.__cachey.extend(y)
        self.__count += len(x)
        if len(self.__cachex) > self.__update_after:
            super(DynamicPlot,self).extend(self.__cachex, self.__cachey, trace=trace)
            self.__cachex.clear()          
            self.__cachey.clear()

    def append(self, x=None, y=None, trace=0):
        """ Update this plot with new data (appends to the end of the trace). If x is left as None, the trace size is used.

        Args:
            x ((int, float), optional): x data. Defaults to current trace size.
            y ((int, float)): y data.
            trace (int, optional): to update. Defaults to 0.
        """
        assert x is not None
        assert trace == 0 # multiple traces not supported yet

        if y is None:
            y = x
            x = self.__count

        self.__cachex.append(x)
        self.__cachey.append(y)
        self.__count += 1

        if len(self.__cachex) > self.__update_after:
            super(DynamicPlot,self).extend(self.__cachex, self.__cachey, trace=trace)
            self.__cachex.clear()          
            self.__cachey.clear()

    def update(self, x=None, y=None, trace=0):
        self.__cachex.clear()
        self.__cachey.clear()
        self.__count = len(x)
        super(DynamicPlot, self).update(x=x,y=y,trace=trace)

# =========== QUIVER ========== #
# TODO move to new another file?

from plotly.figure_factory._quiver import _Quiver

class Quiver:

    def __init__(self, x, y, u, v, scale=0.1, arrow_scale=0.3, angle=math.pi / 9, scaleratio=None, **kwargs):
        super(Quiver, self).__init__()

        if scaleratio is None:
            self.quiver_obj = _Quiver(x, y, u, v, scale, arrow_scale, angle)
        else:
            self.quiver_obj = _Quiver(x, y, u, v, scale, arrow_scale, angle, scaleratio)
        barb_x, barb_y = self.quiver_obj.get_barbs()
        arrow_x, arrow_y = self.quiver_obj.get_quiver_arrows()

        plt = go.Scatter(x=barb_x + arrow_x, y=barb_y + arrow_y, mode="lines", **kwargs)
        self.fig = go.FigureWidget(go.Figure(data=[plt], layout=vis_plot.layout_default()))

    def display(self):
        display(self.fig)

    def update(self, x, y, u, v):
        self.quiver_obj.x = x
        self.quiver_obj.y = y
        self.quiver_obj.u = u
        self.quiver_obj.v = v
        barb_x, barb_y = self.quiver_obj.get_barbs()
        arrow_x, arrow_y = self.quiver_obj.get_quiver_arrows()
        self.fig.data[0].x = barb_x + arrow_x
        self.fig.data[0].y = barb_y + arrow_y

    def __str__(self):
        return str(self.fig)

    def __repr__(self):
        return str(self.fig)

class QuiverUnit(Quiver):

    def __init__(self, u, v, **kwargs):
        x = y = np.zeros_like(u)
        super(QuiverUnit, self).__init__(x,y,u,v,**kwargs)

    def update(self, u,v):
        x = y = np.zeros_like(u)
        return super(QuiverUnit, self).update(x,y,u,v)

def quiver(x=[], y=[], u=[], v=[], arrow_scale=.1, scaleratio=1, **kwargs):
    return Quiver(x,y,u,v,arrow_scale=arrow_scale, scaleratio=scaleratio, **kwargs)

def quiver_unit(u=[], v=[], scale=1., arrow_scale=.1, scaleratio=1, **kwargs):
    return QuiverUnit(u,v,arrow_scale=arrow_scale, scaleratio=scaleratio, **kwargs)

# ========================================= #

def histogram(x, bins=20, legend=None, log_scale=False, show=True):
    fig = vis_plot.histogram(x, bins=bins, legend=legend, log_scale=log_scale, show=False)
    fig = go.FigureWidget(fig)
    if show:
        display(fig)
    return fig

def plot(x=[],y=None,mode=line_mode.line,legend=None,show=True):

    plot = SimplePlot(x,y,mode=mode,legend=legend)
    if show:
        plot.display()
    return plot

def scatter(x,y,mode=line_mode.marker,legend=None,show=True):
    plot = SimplePlot(x,y,mode=mode,legend=legend)
    if show:
        plot.display()
    return plot

def dynamic_plot(x=[],y=[], update_after=100, mode=line_mode.line,legend=None,show=True):
    plot = DynamicPlot(x=x,y=y, update_after=update_after, mode=mode,legend=legend)
    if show:
        plot.display()
    return plot

def progress(iterator, length=None, info=None):
    if info is not None:
        print(info)
    
    if length is None:
        try:
            length = len(iterator)
        except:
            print("Failed determin length of iterator, progress bar failed to display. Please provide the 'length' argument.")
            for i in iterator:
                yield i
            return
    
    f = IntProgress(min=0, max=length, step=1, value=0) # instantiate the bar
    display(f)

    for i in iterator:
        yield i
        f.value += 1


    
def scatter_image(x, y, images, scale=1, mode=line_mode.both, scatter_colour=None, line_colour=None, width=None, height=None):
    #images must be in NHWC format
    if transform.isCHW(images):
        images = transform.HWC(images)

    if transform.is_float(images):
        images = transform.to_integer(images)

    fig = go.FigureWidget(data=[dict(type='scattergl',x=x, y=y,
                mode=mode,
                marker=dict(color=scatter_colour),
                line=dict(color=line_colour))])
    fig.layout.hovermode = 'closest'
    scatter = fig.data[0]

    #convert images to png format
    image_width = '{0}px'.format(int(images.shape[2] * scale))
    image_height = '{0}px'.format(int(images.shape[1] * scale))
    images = [transform.to_bytes(image) for image in images]

    image_widget = Image(value=images[0], layout=Layout(height=image_height, width=image_width))
    
    def hover_fn(trace, points, state):
        ind = points.point_inds[0]
        image_widget.value = images[ind]

    scatter.on_hover(hover_fn)
    #fig.show()
    #print("WHAT")

    #box_layout = widgets.Layout(display='flex',flex_flow='row',align_items='center',width='100%',height='100%')
    #display(HBox([fig, image_widget], layout=box_layout)) #basically... this needs to be done in jupyter..?!]

    return fig, image_widget


class SimpleImage:

    def __init__(self, image, scale=1, interpolation=transform.interpolation.nearest):
        self.__scale = scale
        self.__interpolation = interpolation
        self.__image = self.transform(image)

        self.__canvas = Canvas(width=self.__image.shape[1], height=self.__image.shape[0], scale=1)
        self.__canvas.put_image_data(self.__image, 0, 0) 
        
    @property
    def fig(self):
        return self.__canvas

    def transform(self, image):
        if transform.isCHW(image) and not transform.isHWC(image):
            image = transform.HWC(image) #transform to HWC format for display...
        elif not transform.isHWC(image):
            raise ValueError("Argument: \"image\" must be in HWC format")

        if transform.is_integer(image):
            image = transform.to_float(image)
        else:
            image = image.astype(np.float32) #must be float32...

        if image.shape[-1] != 3:
            image = transform.colour(image, components=(1,1,1)) #requires HWC float format...
        if self.__scale != 1:
            image = transform.scale(image, self.__scale, interpolation=self.__interpolation)

        return transform.to_integer(image) #finally transform to int

    def display(self):
        box_layout = widgets.Layout(display='flex',flex_flow='row',align_items='center',width='100%')
        display(HBox([self.__canvas], layout=box_layout))

    def update(self, image):
        self.set_image(image)

    def set_image(self, image):
        #TODO if this is live there might be problems... give the option of preprocessing via transform

        image = self.transform(image)
        assert image.shape == self.__image.shape #shapes must be equal after scaling
        self.__image = image
        self.__canvas.put_image_data(image)

    @property
    def widget(self):
        return self.__canvas

    def scale(self):
        raise NotImplementedError("TODO scale the image?")

def layout_horizontal(*figs):
    box_layout = widgets.Layout(display='flex',flex_flow='row',align_items='center',width='100%')
    return HBox(figs, layout=box_layout)

def layout_vertical(*figs):
    box_layout = widgets.Layout(display='flex',flex_flow='col',align_items='center',width='100%')
    return VBox(figs, layout=box_layout)

def text(value='text', size=1):
    return HTML("<h{1}>{0}</h{1}>".format(value, str(size)))

def image(image, scale=1, interpolation=transform.interpolation.nearest, show=True):
    image_widget = SimpleImage(image, scale=scale, interpolation=interpolation)
    if show:
        image_widget.display()
    return image_widget

def images(images, scale=1, on_interact=lambda x: None, step=1, value=0, window=1, interpolation=transform.interpolation.nearest, show=True):
    image_widget = SimpleImage(images[0], scale=scale, interpolation=interpolation)
    
    if hasattr(on_interact, '__getitem__'):
        l = on_interact
        def list_on_interact(z):
            print("value:", l[z]) #this only works with later version of ipython?
        on_interact = list_on_interact

    def slide(x):
        image_widget.set_image(images[x])
        on_interact(x)

    interact(slide, x=IntSlider(min=0, max=len(images)-1, step=step, value=value, layout=dict(width='99%'))) #width 100% makes a scroll bar appear...?
    if show:
        image_widget.display()
    return image_widget



def mouse_hover(widget, callback):
    e = Event(source=widget, watched_events=['mousemove'])
    e.on_dom_event(callback)

class __IPyCanvasMouseMoveHandler:

    def __init__(self, canvas, callback, scale=1):
        self.px = self.py = 0
        self.callback = callback
        self.scale = scale
        self.canvas = canvas

    def __call__(self, x, y):
        x = min(max(int(x / self.scale), 0), int(self.canvas.width / self.scale -1))
        y = min(max(int(y / self.scale), 0), int(self.canvas.height / self.scale -1))
        if self.px == x and self.py == y:
            return
        self.callback(x, y)

class __GridSnap:

    def __init__(self, snap=(1,1)):
        self._snapx = snap[0]
        self._snapy = snap[1]
    
    def snap(self, x, y):
        x = int(x / self._snapx) * self._snapx
        y = int(y / self._snapy) * self._snapy

class __IPyEventMouseMoveHandler:

    def __init__(self, widget, callback, snap=(1,1), min_position=None, max_position=None):
        self.px = self.py = 0
        self._snapx, self._snapy = snap

        self.callback = callback
        self.widget = widget

        if max_position is None:
            self._maxx, self._maxy = widget.width, widget.height
        else:
            self._maxx, self._maxy = max_position

        if min_position is None:
            self._minx = self._miny = 0
        else:
            self._minx, self._miny = min_position

        mouse_hover(widget, self)

    def __call__(self, event):
        x, y = event['relativeX'] - 2, event['relativeY'] - 2
        x = min(max(int(x / self._snapx) * self._snapx, self._minx), self._maxx)
        y = min(max(int(y / self._snapy) * self._snapy, self._miny), self._maxy)

        if self.px == x and self.py == y:
            return

        self.px, self.py = x, y
        self.callback(x, y)

    @property
    def widget_width(self):
        return int(self.widget.width)
    
    @property
    def widget_height(self):
        return int(self.widget.height)

def image_roi(image, callback=lambda *_: None, box_shape=(4,4), snap=(1,1), scale=1, highlight_alpha=0.2, show=True):
    assert transform.isHWC(image)

    if transform.is_integer(image):
        image = transform.to_float(image)
    else:
        image = image.astype(np.float32) #must be float32...

    image = transform.colour(image) #requires HWC float format...
    image = transform.scale(image, scale, interpolation=transform.interpolation.nearest)
    
    image = transform.to_integer(image)

    canvas = MultiCanvas(2, width=image.shape[0], height=image.shape[1], scale=1)
    canvas[0].put_image_data(image, 0, 0)
    
    bw = box_shape[0] * scale
    bh = box_shape[1] * scale

    out = Output() #for printing stuff..

    @out.capture()
    def draw_callback(x, y):
        canvas[1].clear()
        canvas[1].fill_style = 'white'
        canvas[1].global_alpha = highlight_alpha
        canvas[1].fill_rect(x,y,bw,bh)
    
        canvas[1].global_alpha = 1.
        canvas[1].stroke_style = 'red'
        canvas[1].stroke_rect(x,y,bw,bh)

        callback(x,y)
    
    snap = (snap[0] * scale, snap[1] * scale)
    max_position = (canvas.width - bw, canvas.height - bh)
    mmh = __IPyEventMouseMoveHandler(canvas, draw_callback, snap=snap, max_position=max_position)
    if show:
        display(VBox([canvas,out]))

    return canvas, mmh
