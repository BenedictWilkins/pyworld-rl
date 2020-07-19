import plotly
import plotly.graph_objects as go
import plotly.subplots as subplots

import numpy as np
import io

from types import SimpleNamespace as Namespace

try:
    from PIL import Image
except:
    pass

try:
    import cv2
except:
    pass

from ipywidgets.embed import embed_minimal_html
import webbrowser
import os



class SimplePlotException(Exception):
    pass

class PlotLayoutException(Exception):
    pass

class Singleton:

    def __init__(self, value):
        self.value = value
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.value
    
    def __getitem__(self, key):
        return self.value

line_mode = Namespace(line='lines', marker='markers', both='lines+markers')

def __layout__(): #default plot layout
    return {'plot_bgcolor':'white',
            'xaxis':{'ticks':'outside', 'showgrid':False, 'showline':True, 'mirror':True, 'linewidth':2, 'linecolor':'black'},
            'yaxis':{'ticks':'outside', 'showgrid':False, 'showline':True, 'mirror':True, 'linewidth':2, 'linecolor':'black'}}


def __layout_noaxis__(): #plot layout without axes
    return {'plot_bgcolor':'white',
            'xaxis':{'ticks':'', 'showgrid':False, 'showline':False, 'showticklabels':False},
            'yaxis':{'ticks':'', 'showgrid':False, 'showline':False, 'showticklabels':False}}

def layout_default():
    return __layout__()

def layout_noaxis():
    return __layout_noaxis__()

def __line_mode__(mode):
    assert mode in line_mode.__dict__.values()
    if not isinstance(mode, (list, tuple)):
        mode = Singleton(mode)
    return mode

def __listdepth__(x): #get depth of a nested list
    if hasattr(x, "__getitem__"):
        try:
            return __listdepth__(x[0]) + 1
        except:
            pass
        return 0

def __legend__(legend, size): #default legend
    if legend is not None:
        assert isinstance(legend, (list, tuple))
        return legend
    else:
        return range(1, size + 1)

class SimplePlot:
    
    def __init__(self, x, y, legend=None, mode=line_mode.marker, **kwargs):
        if y is None:
            raise SimplePlotException("argument y cannot be None.")
        
        x, y = self.__validate__(x, y)

        mode = __line_mode__(mode)
        legend = __legend__(legend, len(x))
        layout = __layout__()

        fig = go.Figure(layout=layout)

        for i, xi, yi in zip(range(len(x)), x, y):
            fig.add_trace(go.Scattergl(x=xi, y=yi, mode=mode[i], name=legend[i]))

        fig.update_layout(plot_bgcolor='white', margin=dict(t=10,l=10,r=10,b=10))
        self.fig = fig

    def length(self, trace=0):
        return len(self.fig.data[trace].x)

    def display(self):
        self.fig.show()

    def add_shape(self, *args, **kwargs):
        self.fig.add_shape(*args, **kwargs)

    def add_hline(self, y):
        shapes = list(self.fig.layout.shapes)
        shapes.append(dict(type= 'line', yref= 'y', y0=y, y1=y, xref= 'paper', x0=0, x1=1))
        self.fig.layout.shapes = shapes

    def add_vline(self, x):
        shapes = list(self.fig.layout.shapes)
        shapes.append(dict(type= 'line', yref= 'paper', y0= 0, y1=1, xref= 'x', x0=x, x1=x))
        self.fig.layout.shapes = shapes

    def trace(self, x, y, **kwargs):
        x, y = self.__validate__(x, y)
        for i, xi, yi in zip(range(len(x)), x, y):
            self.fig.add_trace(go.Scatter(x=xi, y=yi, **kwargs))

    def point(self, x, y, **kwargs):
        kwargs['mode'] = line_mode.marker
        self.fig.add_trace(go.Scatter(x=[x], y=[y], **kwargs))
    
    def append(self, x, y, trace=0):
        self.extend(x,y,trace=trace)

    def extend(self, x, y, trace=0):
        assert isinstance(self.fig.data[trace].x, (list, tuple)) and isinstance(self.fig.data[trace].y, (list,tuple))
        assert len(x) == len(y)
        self.fig.data[trace].x += tuple(x)
        self.fig.data[trace].y += tuple(y)

    def update(self, x, y, trace=0):
        self.set_data(x, y, trace=trace)       

    def set_data(self, x, y, trace=0):
        self.fig.data[trace].x = x
        self.fig.data[trace].y = y

    def __validate__(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape:
                raise SimplePlotException("Invalid shapes: {0},{1} for x,y arguments: shapes must be equal".format(x.shape, y.shape))
            if len(x.shape) == 1:
                # a single trace has been provided
                x = x[np.newaxis, :]
                y = y[np.newaxis, :]
                if len(x.shape) != 2:
                    raise SimplePlotException("Invalid shapes: {0},{1} for x,y arguments: shapes must be in (B,N) format,".format(x.shape,y.shape)
                                                + "\n where B is the number of traces and N is the size of each trace.")
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            #multiple traces have been given
            xdepth = __listdepth__(x)
            ydepth = __listdepth__(y)
            if xdepth != ydepth:
                raise SimplePlotException("Invalid list depth: {0},{1} for x,y arguments: depths must be equal.".format(xdepth,ydepth))
            if not (0 <= xdepth <= 2):
                raise SimplePlotException("Invalid list depth: {0},{1} for x,y arguments: depths must be 1 or 2.".format(xdepth, ydepth))
            if xdepth < 2:
                # a single trace has been given
                x = [x]
                y = [y]
        return x,y



    
def show_widget(widget, title="widget"): #???
    path = os.getcwd() + ".plot/"
    if not os.path.exists(path):
        os.makedirs(path)
    file = path + "temp.html"
    embed_minimal_html(file, views=[widget], title=title)
    webbrowser.open_new_tab(file)

def plot_coloured(x, y, z, bins=10, fig=None, row=None, col=None, show=True): #TODO rename this...?
    '''
        Plots x and y with lines coloured according to z. 
        Arguments:
            x: to plot
            y: to plot
            z: line colour values
    '''
    layout = __layout__()
    if isinstance(bins, int):
        min_z, max_z = np.min(z), np.max(z)
        bins = np.linspace(min_z, max_z, bins)

    #TODO when plotly allows for line segments to be coloured, fix this stuff!
    if fig is None:
        fig = go.Figure()

    fig.add_trace(go.Scatter(x=x,y=y, mode='markers+lines', marker=dict(color=z)), row=row, col=col)

    #for i in range(x.shape[0]):
    #    fig.add_trace(go.Scatter(x=x[i:i+2], y=y[i:i+2], mode='lines', line = dict(color='rgb({0},0,0)'.format(np.random.randint(0,255)))))
    
    if show:
        fig.show()
    return fig





def plot(x, y, mode = line_mode.line, legend=None, show=True):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        assert x.shape == y.shape
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        xdepth = __listdepth__(x)
        ydepth = __listdepth__(y)
        assert xdepth == ydepth
        assert 0 < xdepth < 3
        if xdepth == 1:
            x = [x]
            y = [y]
    
    if not isinstance(mode, (list, tuple)):
        mode = Singleton(mode)

    legend = __legend__(legend, len(x))
    layout = __layout__()

    fig = go.Figure(layout=layout)

    for i, xi, yi in zip(range(len(x)), x, y):
        fig.add_trace(go.Scatter(x=xi, y=yi, mode=mode[i], name=legend[i]))

    fig.update_layout(plot_bgcolor='white')

    if show:
        fig.show()

    return fig

def histogram(x, bins=20, legend=None, log_scale=False, fig=None, row=None, col=None, show=True):
    if isinstance(x, np.ndarray):
        x = x.squeeze()
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
    elif isinstance(x, (list, tuple)):
        xdepth = __listdepth__(x)
        assert 0 < xdepth < 3
        if xdepth == 1:
            x = [x]

    binsize = max([max(z) - min(z) for z in x]) / bins

    legend = __legend__(legend, len(x))
    layout = __layout__()

    if fig is None:
        fig = go.Figure(layout=layout)
        fig.update_layout(plot_bgcolor='white')

    if log_scale:
        fig.update_layout(yaxis_type="log")

    for i, xi in zip(range(len(x)), x):
        fig.add_trace(go.Histogram(x=xi, name=legend[i], xbins=dict(size=binsize)), row=row, col=col)
    if show:
        fig.show()
    return fig

def histograms(*x, bins=20, legend=None, log_scale=False, rows=None, cols=None, show=True):
    subplots.make_subplots(rows=rows, cols=cols)



def histogram_slider(fig, sizes=range(1,20,1)):
    fig = go.FigureWidget(fig)
    fig.layout.sliders = [dict(
                    active = 10,
                    currentvalue = {"prefix": "bin size: "},
                    pad = {"t": 20},
                    steps = [dict(label = str(i), method = 'restyle',  args = ['xbins.size', i]) for i in sizes]
                )]
    return fig

def plot_image(images, show=True):
    if len(images.shape) == 4:
        pass
    elif len(images.shape) == 3:
        images = images[np.newaxis,...]
    elif len(images.shape) == 2:
        images = images[np.newaxis,:,:,np.newaxis]

    g = int(np.ceil(np.sqrt(images.shape[0])))
    fig = subplots.make_subplots(rows=g, cols=g, 
                    shared_xaxes=True, shared_yaxes=True, 
                    vertical_spacing=0.004, horizontal_spacing=0.004)
    fig.update_xaxes(**__layout_noaxis__()['xaxis'])
    fig.update_yaxes(**__layout_noaxis__()['yaxis'])
    
    for i, image in enumerate(images, 0):
        fig.add_trace(go.Image(z = image), (i % g) + 1, (i // g) + 1)

    if show:
        fig.show()

def scroll_images(images, show=True):
    if len(images.shape) == 4:
        pass
    elif len(images.shape) == 3:
        images = images[..., np.newaxis]
    
    #build each trace
    data = [go.Image(z = image, visible = False) for image in images]
    data[0]['visible'] = True
    steps = [dict(method='update', args=[{'visible': [t == i for t in range(len(data))]}]) for i in range(len(images))]

    fig = go.Figure(data=data, layout=__layout_noaxis__())

    fig = go.FigureWidget(fig)
    fig.layout.sliders = [dict(
                    active = 0,
                    currentvalue = {"prefix": "image : "},
                    pad = {"t": 20},
                    steps = steps
                )]

    if show:
        fig.show()

    return fig

def to_numpy(fig, scale=0.25):
    return np.array(Image.open(io.BytesIO(fig.to_image(".png", scale=0.25))))

def __frame_args__(duration):  
    return {"frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"}}

def __play__(duration):
    return {"args": [None, __frame_args__(50)],"label": "&#9654;", "method": "animate"}

def __pause__():
    return {"args": [[None], __frame_args__(0)],"label": "&#9724;","method": "animate"}

def play(images, show=True):
    if len(images.shape) == 4:
        pass
    elif len(images.shape) == 3:
        images = images[..., np.newaxis]

    layout = __layout_noaxis__()

    frames = [go.Frame(data=go.Image(z = images[i]), name=str(i)) for i in range(len(images))]
    fig = go.Figure(data=go.Image(z = images[0]), frames=frames, layout=layout)

    updatemenus = [{"buttons": [__play__(50), __pause__()],
                    "direction": "left", "pad": {"r": 10, "t": 70},"type": "buttons","x": 0.1,"y": 0}]

    sliders = [{"pad": {"b": 10, "t": 60},
                "len": 0.9, "x": 0.1, "y": 0,
                "steps": [{"args": [[f.name], __frame_args__(0)],
                        "label": str(k),"method": "animate"}
                for k, f in enumerate(fig.frames)]}]

    fig.update_layout(updatemenus=updatemenus, sliders=sliders)
    
    if show:
        fig.show()
    return fig

def subplot(rows=1, cols=1):
    return subplots.make_subplots(rows=rows, cols=cols)

'''
def subplot(*figures, show=True):
    #layout information is lost...
    size = int(np.ceil(np.sqrt(len(figures))))
    fig = subplots.make_subplots(rows = size, cols = size)
    for i, f in enumerate(figures):
        fd = f.to_dict()
        for d in fd['data']:
            fig.add_trace(d, row=(i % size) +1, col = (i // size) + 1)

    if show:
        fig.show()

    return fig
'''


 
if __name__ == "__main__":

    def histogram_demo():
        y1 = np.random.randint(0,10, size=100000)
        y2 = np.random.randint(0,10, size=100000)
        fig = histogram([y1, y2], log_scale=True, bins=5, show=False)
        fig = histogram_slider(fig, range=range(1,50,3))
        fig.show()

    def histogram_demo2():
        y1 = np.random.randint(0,10, size=(1000,1))
        fig = histogram(y1, log_scale=True, bins=5, show=False)
        fig = histogram_slider(fig, range=range(1,50,3))
        fig.show()

    def image_demo():
        image = np.random.randint(0,255, size=(500,100,100,3))
        #plot_image(image)
        play(image)

    def subplot_demo():
        y1 = np.random.randint(0,10, size=10)
        y2 = np.random.randint(0,10, size=10)
        fig = histogram(y1, log_scale=True, bins=5, show=False)
        #from pprint import pprint
        #pprint(fig.to_dict())
        fig = subplot(fig)
        fig.show()

    def plot_coloured_demo():
        x = np.linspace(0, 4*np.pi, 1000)
        y = np.sin(x)
        z = np.linspace(0,10,1000)
        plot_coloured(x,y,z)

    #http://etpinard.xyz/plotly-dashboards/hover-images/
    #https://github.com/etpinard/plotly-dashboards/blob/master/hover-images/index.html