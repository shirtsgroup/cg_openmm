
import signac
import h5py
from flow import FlowProject
import os
from simtk import unit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# This script extracts heat capacity data from signac jobs, and plots the C_v vs. T data series.

proj_directory = os.getcwd()

project = signac.get_project()

theta = []
alpha = []

njobs = 0

C_v = {}
dC_v = {}
T = {}
for job in project:
    with job.data:
        if 'C_v' in job.data:
            njobs += 1
            theta.append(job.statepoint['equil_bond_angle_bb_bb_bb'])
            alpha.append(job.statepoint['equil_torsion_angle_bb_bb_bb_bb'])
            
            C_v[f'theta_{theta[-1]}_alpha_{alpha[-1]}'] = job.data['C_v'][:]
            dC_v[f'theta_{theta[-1]}_alpha_{alpha[-1]}'] = job.data['dC_v'][:]
            T[f'theta_{theta[-1]}_alpha_{alpha[-1]}'] = job.data['T_list_C_v'][:]                       
                  
# Sort the data by theta, then alpha
unsorted_zip = zip(theta, alpha)
sorted_tuple = sorted(unsorted_zip, key=lambda x: (x[0], x[1]))
sorted_array = np.asarray(sorted_tuple) 

theta = sorted_array[:,0]
alpha = sorted_array[:,1]                 
                  
# Plot the full C_v vs T curves
series_per_page = 9

nmax = series_per_page

with PdfPages('cv_data_combined_theta_vs_alpha.pdf') as pdf:
    page_num=1
    plotted_per_page=0
    total_plotted = 0
    plt.figure() 
    
    for i in range(len(theta)):
        if plotted_per_page <= nmax:
            plt.plot(T[f'theta_{theta[i]}_alpha_{alpha[i]}'],C_v[f'theta_{theta[i]}_alpha_{alpha[i]}'],'-',
                label=f"theta = {theta[i]}, alpha = {alpha[i]}")
                
            plotted_per_page += 1
            total_plotted += 1
            
        if (plotted_per_page >= nmax) or (total_plotted==njobs):
           # Save and close previous page
            plt.legend(loc='upper center',fontsize=6)            
            plt.xlim(200,500)
            plt.ylim(0,4)            
            plt.xlabel('T')
            plt.ylabel('C_v (T)')

            # Save current figure to pdf page:
            pdf.savefig(bbox_inches='tight')   
            plt.close()
            plotted_per_page = 0
            page_num += 1    
            
            
# Sort the data by alpha, then theta
unsorted_zip = zip(theta, alpha)
sorted_tuple = sorted(unsorted_zip, key=lambda x: (x[1], x[0]))
sorted_array = np.asarray(sorted_tuple) 

theta = sorted_array[:,0]
alpha = sorted_array[:,1]                 
                  
# Plot the full C_v vs T curves
series_per_page = 8

nmax = series_per_page

with PdfPages('cv_data_combined_alpha_vs_theta.pdf') as pdf:
    page_num=1
    plotted_per_page=0
    total_plotted = 0
    plt.figure() 
    
    for i in range(len(theta)):
        if plotted_per_page <= nmax:
            plt.plot(T[f'theta_{theta[i]}_alpha_{alpha[i]}'],C_v[f'theta_{theta[i]}_alpha_{alpha[i]}'],'-',
                label=f"theta = {theta[i]}, alpha = {alpha[i]}")
                
            plotted_per_page += 1
            total_plotted += 1
            
        if (plotted_per_page >= nmax) or (total_plotted==njobs):
           # Save and close previous page
            plt.legend(loc='upper center',fontsize=6)            
            plt.xlim(200,500)
            plt.ylim(0,4)            
            plt.xlabel('T')
            plt.ylabel('C_v (T)')

            # Save current figure to pdf page:
            pdf.savefig(bbox_inches='tight')   
            plt.close()
            plotted_per_page = 0
            page_num += 1                
            