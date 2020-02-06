import plotly.graph_objects as go
import plotly.subplots as subplots

import numpy as np

from types import SimpleNamespace as Namespace

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

def __listdepth__(x):
    try:
        return __listdepth__(x[0]) + 1
    except:
        return 0

def __legend__(legend, size):
    if legend is not None:
        assert isinstance(legend, (list, tuple))
        return legend
    else:
        return range(1, size + 1)

def __layout__():
    return {'xaxis':{'ticks':'outside', 'showgrid':False, 'showline':True, 'mirror':True, 'linewidth':2, 'linecolor':'black'},
            'yaxis':{'ticks':'outside', 'showgrid':False, 'showline':True, 'mirror':True, 'linewidth':2, 'linecolor':'black'}}

def __layout_noaxis__():
    return {'xaxis':{'ticks':'', 'showgrid':False, 'showline':False, 'showticklabels':False},
            'yaxis':{'ticks':'', 'showgrid':False, 'showline':False, 'showticklabels':False}}

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


def histogram(x, bins=20, legend=None, log_scale=False, show=True):
    if isinstance(x, np.ndarray):
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

    fig = go.Figure(layout=layout)
    fig.update_layout(plot_bgcolor='white')
    if log_scale:
        fig.update_layout(yaxis_type="log")

    for i, xi in zip(range(len(x)), x):
        fig.add_trace(go.Histogram(x=xi, name=legend[i], xbins=dict(size=binsize)))
    if show:
        fig.show()

    return fig

def histogram_slider(fig, range=range(1,20,1)):
    fig = go.FigureWidget(fig)
    fig.layout.sliders = [dict(
                    active = 10,
                    currentvalue = {"prefix": "bin size: "},
                    pad = {"t": 20},
                    steps = [dict(label = i, method = 'restyle',  args = ['xbins.size', i]) for i in range]
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

if __name__ == "__main__":

    def histogram_demo():
        y1 = np.random.randint(0,10, size=100000)
        y2 = np.random.randint(0,10, size=100000)
        fig = histogram([y1, y2], log_scale=True, bins=5, show=False)
        fig = histogram_slider(fig, range=range(1,50,3))
        fig.show()

    def image_demo():
        image = np.random.randint(0,255, size=(500,100,100,3))
        #plot_image(image)
        play(image)

    histogram_demo()