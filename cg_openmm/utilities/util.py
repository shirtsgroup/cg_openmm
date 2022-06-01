import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app.element as elem
from openmm import *
from openmm import unit
from openmm.app import *
from scipy.optimize import curve_fit


def distance(positions_1, positions_2):
    """
    Calculate the distance between two particles, given their positions.

    :param positions_1: Positions for the first particle
    :type positions_1: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [3] ), simtk.unit )

    :param positions_2: Positions for the first particle
    :type positions_2: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [3] ), simtk.unit )

    :returns:
        - distance ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_) - Distance between two particles

    :Example:

    >>> from foldamers.cg_model.cgmodel import CGModel
    >>> cgmodel = CGModel()
    >>> particle_1_coordinates = cgmodel.positions[0]
    >>> particle_2_coordinates = cgmodel.positions[1]
    >>> particle_distance = distance(particle_1_coordinates,particle_2_coordinates)

    """

    distance = np.sqrt(np.sum(np.power((positions_1 - positions_2),2)))

    return distance


def get_box_vectors(box_size):
    """
    Given a simulation box length, construct a vector.

    :param box_size: Length of individual sides of a simulation box
    :type box_size: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( float, simtk.unit )

    :returns:
         - box_vectors ( List( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) ) - Vectors to use when defining an OpenMM simulation box.

    """

    units = box_size.unit
    a = unit.Quantity(np.zeros([3]), units)
    a[0] = box_size
    b = unit.Quantity(np.zeros([3]), units)
    b[1] = box_size
    c = unit.Quantity(np.zeros([3]), units)
    c[2] = box_size
    box_vectors = [a, b, c]
    return box_vectors


def set_box_vectors(system, box_size):
    """
    Impose a set of simulation box vectors on an OpenMM simulation object.

    :param system: OpenMM System()
    :type system: `System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_

    :param box_size: Length of individual sides of a simulation box
    :type box_size: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( float, simtk.unit )

    :returns:
        - system (`System() <https://simtk.org/api_docs/openmm/api4_1/python/classsimtk_1_1openmm_1_1openmm_1_1System.html>`_) - OpenMM system object

    """

    a, b, c = get_box_vectors(box_size)
    system.setDefaultPeriodicBoxVectors(a, b, c)
    return system


def lj_v(positions_1, positions_2, sigma, epsilon, r_exp=12.0, a_exp=6.0):
    """
    Calculate the Lennard-Jones interaction energy between two particles, given their positions and definitions for their equilbrium interaction distance (sigma) and strength (epsilon).

    :param positions_1: Positions for the first particle
    :type positions_1: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [3] ), simtk.unit )

    :param positions_2: Positions for the first particle
    :type positions_2: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ( np.array( [3] ), simtk.unit )

    :param sigma: Lennard-Jones equilibrium interaction distance for two non-bonded particles
    :type sigma: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

    :param epsilon: Lennard-Jones equilibrium interaction energy for two non-bonded particles.
    :type epsilon: `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_

    :param r_exp: repulsive exponent (default=12.0)
    :type r_exp: float
    
    :param a_exp: attractive exponent (default=6.0)
    :type a_exp: float

    :returns:
       - v ( `Quantity() <http://docs.openmm.org/development/api-python/generated/simtk.unit.quantity.Quantity.html>`_ ) - Lennard-Jones interaction energy

    """

    dist = distance(positions_1, positions_2)

    if r_exp == 12.0 and a_exp == 6.0:
        # This is a standard LJ 12-6 function
        v = 4*epsilon*(np.power((sigma/dist),12.0)-np.power((sigma/dist),6.0))
    else:
        # This is a generalized LJ (Mie) function
        C = (r_exp/(r_exp-a_exp))*(r_exp/a_exp)**(m/(r_exp-a_exp))
        v = C*epsilon*(np.power((sigma/dist),r_exp)-np.power((sigma/dist),a_exp))
    
    return v
   
    
def fit_sigmoid(xdata, ydata, plotfile='Q_vs_T_fit.pdf', xlabel='T (K)', ylabel='Q'):
    """
    Fit a sigmoidal curve (such as native contact fraction vs T) to hyperbolic tangent switching function
    
    :param xdata: x data series
    :type xdata: Quantity or numpy 1D array
    
    :param ydata: y data series
    :type ydata: Quantity or numpy 1D array
    
    :param plotfile: Path to output file for plotting results (default='Q_vs_T_fit.pdf')
    :type plotfile: str
    
    :returns:
        - param_opt ( 1D numpy array ) - optimized sigmoid parameters (x0, y0, y1, d) 
        - param_cov ( 2D numpy array ) - estimated covariance of param_opt

    """
    
    # Strip units off quantities:
    if type(xdata[0]) == unit.quantity.Quantity:
        xdata_val = np.zeros((len(xdata)))
        xunit = xdata[0].unit
        for i in range(len(xdata)):
            xdata_val[i] = xdata[i].value_in_unit(xunit)
        xdata = xdata_val
    
    if type(ydata[0]) == unit.quantity.Quantity:
        ydata_val = np.zeros((len(ydata)))
        yunit = ydata[0].unit
        for i in range(len(ydata)):
            ydata_val[i] = ydata[i].value_in_unit(yunit)
        ydata = ydata_val
        
    
    def tanh_switch(x,x0,y0,y1,d):
        return (y0+y1)/2-((y0-y1)/2)*np.tanh(np.radians(x-x0)/d)
        
    param_guess = [np.mean(xdata),np.min(ydata),np.max(ydata),(np.max(xdata)-np.min(xdata))/10]
    
    if np.max(ydata) > 1:
        # This is not native contact fraction data (for example, radius of gyration data)
        bounds = (
            [np.min(xdata), 0, 0, 0],
            [np.max(xdata), np.max(ydata), 3*np.max(ydata), (np.max(xdata)-np.min(xdata))]
        )
    
    else:
        # This is likely native contact fraction data
        bounds = (
            [np.min(xdata), 0, 0, 0],
            [np.max(xdata), 1, 1, (np.max(xdata)-np.min(xdata))]
        )
    
    param_opt, param_cov = curve_fit(tanh_switch, xdata, ydata, param_guess, bounds=bounds)
    
    if plotfile is not None:
        figure = plt.figure()
        
        xfit = np.linspace(np.min(xdata),np.max(xdata),1000)
        yfit = tanh_switch(xfit,param_opt[0],param_opt[1],param_opt[2],param_opt[3])
        
        # Value of y at the critical value of x:
        y_x0 = (param_opt[1]+param_opt[2])/2
        y_x0_err = np.sqrt(np.power(param_cov[1,1],2)+np.power(param_cov[2,2],2))
        
        line1 = plt.plot(
            xdata,
            ydata,
            'ok',
            markersize=4,
            fillstyle='none',
            label='simulation data',
        )
        
        line2 = plt.plot(
            xfit,
            yfit,
            '-b',
            label='hyperbolic fit',
        )
        
        
        if np.max(ydata) <= 1:
            plt.ylim((0,1))
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.legend()
        
        xlim = plt.xlim()
        ylim = plt.ylim()
        
        plt.text(
            (xlim[0]+0.05*(xlim[1]-xlim[0])),
            (ylim[0]+0.60*(ylim[1]-ylim[0])),
            r'$y(x)=\frac{y0+y1}{2}-\frac{y0-y1}{2}\left(\tanh\left(\frac{(x-x0)}{d}\right)\right)$',
            {'fontsize': 10},
        )
        
        plt.text(
            (xlim[0]+0.05*(xlim[1]-xlim[0])),
            (ylim[0]+0.25*(ylim[1]-ylim[0])),
            f'x0 = {param_opt[0]:.4e} \u00B1 {param_cov[0,0]:.4e}\n' \
            f'y0 = {param_opt[1]:.4e} \u00B1 {param_cov[1,1]:.4e}\n' \
            f'y1 = {param_opt[2]:.4e} \u00B1 {param_cov[2,2]:.4e}\n' \
            f'd = {param_opt[3]:.4e} \u00B1 {param_cov[3,3]:.4e}\n' \
            f'y(x0) = {y_x0:>.4e} \u00B1 {y_x0_err:.4e}',
            {'fontsize': 10},
            horizontalalignment='left',
        )
        
        plt.savefig(plotfile)
        plt.close()
    
    return param_opt, param_cov
    
    
    
