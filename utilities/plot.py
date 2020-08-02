import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(
    types_dict,
    hist_data,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    figure_title=None,
    file_name=None,
    marker_string='o-k',
    linewidth=1,
    markersize=6,
):
    """
    Plot angle or torsion distribution and save to file.
    
    :param types_dict: dictionary mapping angle/torsion numeric type to strings
    :type types_dict: dict{str(int): series_name, ...}
    
    :param hist_data: dictionary containing histogram data
    :type hist_data: dict{series_name_density: 1D numpy array, series_name_bin_centers: 1D numpy array, ...}
    
    :param xlabel: label for x-axis
    :type x_label: str
    
    :param ylabel: label for y-axis
    :type y_label: str
    
    :param xlim: limits for x-axis
    :type xlim: list[xlo, xhi]
    
    :param ylim: limits for y-axis
    :type ylim: list(ylo, yhi)
    
    :param figure_title: title of overall plot
    :type figure_title: str
    
    :param file name: name of file, excluding pdf extension
    :type file_name: str
    
    :param marker_string: pyplot format string for line type, color, and symbol type (default = 'o-k')
    :type marker_string: str
    
    :param linewidth: width of plotted line (default=1)
    :type linewidth: float
    
    :param markersize: size of plotted markers (default=6 pts)
    :type markersize: float
   
    """
    
    # Determine number of data series:
    nseries = len(types_dict)
    
    # Determine optimal grid format:
    ncolumn = np.floor(np.sqrt(nseries))
    nrow = np.ceil(nseries/ncolumn)
    
    for key,value in types_dict.items():
        plt.subplot(nrow,ncolumn,int(key))
        plt.plot(
            hist_data[f"{value}_bin_centers"],
            hist_data[f"{value}_density"],
            marker_string,
            linewidth=linewidth,
            markersize=markersize,
        )
    
        if xlim != None:
            plt.xlim(xlim[0],xlim[1])
        if ylim != None:
            plt.ylim(ylim[0],ylim[1])
            
        # Use xlabels for bottom row only:
        # nseries-ncol+1, nseries-ncol+2, ... nseries
        if xlabel != None and (int(key) > (nseries-ncolumn)):
            plt.xlabel(xlabel)
            
        # Use ylabels for left column only:
        # 1, 1+ncol, 1+2*ncol, ...
        if ylabel != None and ((int(key)-1)%ncolumn == 0):
            plt.ylabel(ylabel)
        plt.title(f"{types_dict[key]}")

    if figure_title != None:
        plt.suptitle(figure_title)
        
    plt.savefig(f"{file_name}.pdf")
    plt.close()
    
    return
