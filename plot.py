import matplotlib.pyplot as pyplot
import numpy as np


def plot_distribution(
    xdata,
    ydata,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    plot_title=None,
    file_name=None,
    marker_string='o-k',
    linewidth=1,
    markersize=6,
):
    """
    Plot a single distribution and save to file.
    
    :param x_data: x data series 
    :type x_data: 1D-numpy array
    
    :param y_data: y data series 
    :type y_data: 1D-numpy array   
    
    :param xlabel: label for x-axis
    :type x_label: str
    
    :param ylabel: label for y-axis
    :type y_label: str
    
    :param xlim: limits for x-axis
    :type xlim: list[xlo, xhi]
    
    :param ylim: limits for y-axis
    :type ylim: list(ylo, yhi)
    
    :param plot_title: title of plot
    :type plot_title: str
    
    :param file name: name of file, excluding pdf extension
    :type file_name: str
    
    :param marker_string: pyplot format string for line type, color, and symbol type (default = 'o-k')
    :type marker_string: str
    
    :param linewidth: width of plotted line (default=1)
    :type linewidth: float
    
    :param markersize: size of plotted markers (default=6 pts)
    :type markersize: float
   
    """
    
    pyplot.plot(
        xdata,ydata,marker_string,linewidth=linewidth,markersize=markersize
    )
    
    if xlim != None:
        pyplot.xlim(xlim[0],xlim[1])
    if ylim != None:
        pyplot.ylim(ylim[0],ylim[1])
    if xlabel != None:
        pyplot.xlabel(xlabel)
    if ylabel != None:
        pyplot.ylabel(ylabel)
    if plot_title != None:
        pyplot.title(plot_title)
        
    pyplot.savefig(f"{file_name}.pdf")
    pyplot.close()
    
    return
