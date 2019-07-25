import os
import numpy as np
from simtk import unit
from foldamers.src.cg_model.cgmodel import CGModel
from cg_openmm.src.simulation.rep_exch import replica_exchange
from cg_openmm.src.simulation.tools import run_simulation
import pymbar
from pymbar.examples.heat_capacity import heat_capacity

temperature_list = [300.0+i*1.0 for i in range(11)] * unit.kelvin
print_frequency = 1 # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.picosecond # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
kB = 0.008314462  #Boltzmann constant (Gas constant) in kJ/(mol*K)

cg_model = CGModel(include_torsion_forces=False)
for temperature in temperature_list:
  run_simulation(cg_model,os.getcwd(),total_simulation_time,simulation_time_step,temperature,print_frequency)
exit()
replica_energies,reduced_replica_energies,replica_positions,temperature_list = replica_exchange(cg_model.topology,cg_model.system,cg_model.positions,temperature_list=temperature_list,simulation_time_step=simulation_time_step,total_simulation_time=total_simulation_time,test_time_step=False)
print(replica_energies.shape)
exit()
# Make a list of temperatures
min_temp,max_temp=temperature_list[0]._value,temperature_list[-1]._value
temps = []
delta = 1.0
current_temp = min_temp
while (current_temp <= max_temp):
  temps.append(current_temp)
  current_temp = current_temp + delta

temps = np.array([temp for temp in temps])
originalK = len(temperature_list)
K = len(temps)
N_k = [len(replica_energies[replica]) for replica in range(len(replica_energies))]
NumIterations = 1000

Nall_k = np.zeros([K], np.int32) # Number of samples (n) for each state (k) = number of iterations/energies
E_kn = np.zeros([K, NumIterations], np.float64)

for k in range(originalK):
    E_kn[k,0:N_k[k]] = replica_energies[k,0:N_k[k]]
    Nall_k[k] = N_k[k]

beta_k = 1 / (kB * temps)
u_kln = np.zeros([K,K,NumIterations], np.float64) # u_kln is reduced pot. ener. of segment n of temp k evaluated at temp l
E_kn_samp = np.zeros([K,NumIterations], np.float64) # E_kln is reduced pot. ener. of segment n of temp k evaluated at temp l

nBoots_work = 50 + 1 # we add +1 to the bootstrap number, as the zeroth bootstrap sample is the original
ntypes = 3

allCv_expect = np.zeros([K,ntypes,nBoots_work], np.float64)
dCv_expect = np.zeros([K,ntypes],np.float64)
allE_expect = np.zeros([K,nBoots_work], np.float64)
allE2_expect = np.zeros([K,nBoots_work], np.float64)
dE_expect = np.zeros([K],np.float64)

for n in range(nBoots_work):
    if (n > 0):
        print("Bootstrap: %d/%d" % (n,nBoots))
    for k in range(K):
    # resample the results:
        if Nall_k[k] > 0:
            if (n == 0):  # don't randomize the first one
                booti = np.array(range(N_k[k]))
            else:
                booti=np.random.randint(Nall_k[k],size=Nall_k[k])
            E_kn_samp[k,0:Nall_k[k]] = E_kn[k,booti]

    for k in range(K):
        for l in range(K):
            u_kln[k,l,0:Nall_k[k]] = beta_k[l] * E_kn_samp[k,0:Nall_k[k]]

#------------------------------------------------------------------------
# Initialize MBAR
#------------------------------------------------------------------------

# Initialize MBAR with Newton-Raphson
    if (n==0):  # only print this information the first time
        print("")
        print("Initializing MBAR:")
        print("--K = number of Temperatures with data = %d" % (originalK))
        print("--L = number of total Temperatures = %d" % (K))
        print("--N = number of Energies per Temperature = %d" % (np.max(Nall_k)))

    if (n==0):
        initial_f_k = None # start from zero 
    else:
        initial_f_k = mbar.f_k # start from the previous final free energies to speed convergence

    mbar = pymbar.MBAR(u_kln, Nall_k, verbose=False, relative_tolerance=1e-12, initial_f_k=initial_f_k)

    #------------------------------------------------------------------------
    # Compute Expectations for E_kt and E2_kt as E_expect and E2_expect
    #------------------------------------------------------------------------

    print("")
    print("Computing Expectations for E...")
    E_kln = u_kln  # not a copy, we are going to write over it, but we don't need it any more.
    for k in range(K):
        E_kln[:,k,:]*=beta_k[k]**(-1)  # get the 'unreduced' potential -- we can't take differences of reduced potentials because the beta is different; math is much more confusing with derivatives of the reduced potentials.
    results = mbar.computeExpectations(E_kln, state_dependent = True)
    E_expect = results['mu']
    dE_expect = results['sigma']
    allE_expect[:,n] = E_expect[:]

    # expectations for the differences, which we need for numerical derivatives
    results = mbar.computeExpectations(E_kln,output='differences', state_dependent = True)
    DeltaE_expect = results['mu']
    dDeltaE_expect = results['sigma']
    print("Computing Expectations for E^2...")

    results = mbar.computeExpectations(E_kln**2, state_dependent = True)
    E2_expect = results['mu']
    dE2_expect = results['sigma']
    allE2_expect[:,n] = E2_expect[:]

    results = mbar.getFreeEnergyDifferences()
    df_ij = results['Delta_f']
    ddf_ij = results['dDelta_f']

    #------------------------------------------------------------------------
    # Compute Cv for NVT simulations as <E^2> - <E>^2 / (RT^2)
    #------------------------------------------------------------------------

    if (n==0):
        print("")
        print("Computing Heat Capacity as ( <E^2> - <E>^2 ) / ( R*T^2 ) and as d<E>/dT")

    # Problem is that we don't have a good uncertainty estimate for the variance.
    # Try a silly trick: but it doesn't work super well.
    # An estimator of the variance of the standard estimator of th evariance is
    # var(sigma^2) = (sigma^4)*[2/(n-1)+kurt/n]. If we assume the kurtosis is low
    # (which it will be for sufficiently many samples), then we can say that
    # d(sigma^2) = sigma^2 sqrt[2/(n-1)].
    # However, dE_expect**2 is already an estimator of sigma^2/(n-1)
    # Cv = sigma^2/kT^2, so d(Cv) = d(sigma^2)/kT^2 = sigma^2*[sqrt(2/(n-1)]/kT^2
    # we just need an estimate of n-1, but we can try to get that by var(dE)/dE_expect**2
    # it's within 50% or so, but that's not good enough.
    
    allCv_expect[:,0,n] = (E2_expect - (E_expect*E_expect)) / ( kB * Temp_k**2)


    ####################################
    # C_v by fluctuation formula
    ####################################

    #Cv = (A - B^2) / (kT^2)
    # d2(Cv) = [1/(kT^2)]^2 [(dCv/dA)^2*d2A + 2*dCv*(dCv/dA)*(dCv/dB)*dAdB + (dCv/dB)^2*d2B]
    # = [1/(kT^2)]^2 [d2A - 4*B*dAdB + 4*B^2*d2B]
    # But this formula is not working for uncertainies!

    if (n==0):
        N_eff = (E2_expect - (E_expect*E_expect))/dE_expect**2  # sigma^2 / (sigma^2/n) = effective number of samples
        dCv_expect[:,0] = allCv_expect[:,0,n]*np.sqrt(2/N_eff)

    # only loop over the points that will be plotted, not the ones that
    for i in range(originalK, K):

        # Now, calculae heat capacity by T-differences
        im = i-1
        ip = i+1
        if (i==originalK):
            im = originalK
        if (i==K-1):
            ip = i

	####################################
        # C_v by first derivative of energy
        ####################################

        if (dertype == 'temperature'):  # temperature derivative
            # C_v = d<E>/dT
            allCv_expect[i,1,n] = (DeltaE_expect[im,ip])/(Temp_k[ip]-Temp_k[im])
            if (n==0):
                dCv_expect[i,1] = (dDeltaE_expect[im,ip])/(Temp_k[ip]-Temp_k[im])
        elif (dertype == 'beta'):  # beta derivative
            #Cv = d<E>/dT = dbeta/dT d<E>/beta = - kB*T(-2) d<E>/dbeta  = - kB beta^2 d<E>/dbeta
            allCv_expect[i,1,n] = kB * beta_k[i]**2 * (DeltaE_expect[ip,im])/(beta_k[ip]-beta_k[im])
            if (n==0):
                dCv_expect[i,1] = -kB * beta_k[i]**2 *(dDeltaE_expect[ip,im])/(beta_k[ip]-beta_k[im])

	####################################
        # C_v by second derivative of free energy
        ####################################

        if (dertype == 'temperature'):
            # C_v = d<E>/dT = d/dT k_B T^2 df/dT = 2*T*df/dT + T^2*d^2f/dT^2

            if (i==originalK) or (i==K-1):
                 # We can't calculate this, set a number that will be printed as NAN
                 allCv_expect[i,2,n] = -10000000.0
            else:
                allCv_expect[i,2,n] = kB*Temp_k[i]*(2*df_ij[ip,im]/(Temp_k[ip]-Temp_k[im]) +
						    Temp_k[i]*(df_ij[ip,i]-df_ij[i,im])/
						    ((Temp_k[ip]-Temp_k[im])/(ip-im))**2)

            if (n==0):
                # Previous work to calculate the uncertainty commented out, should be cleaned up eventually
                # all_Cv_expect[i,2,n] = kB*Temp_k[i]*(2*df_ij[ip,i]+df_ij[i,im]/(Temp_k[ip]-Temp_k[im]) + Temp_k[i]*(df_ij[ip,i]-df_ij[i,im])/(Temp_k[ip]-Temp_k[i])**2)
                #all_Cv_expect[i,2,n] = kB*([2*Temp_k[i]/(Temp_k[ip]-Temp_k[im]) + Temp_k[i]**2/(Temp_k[ip]-Temp_k[i])**2]*df_ij[ip,i] + [2*Temp_k[i]/(Temp_k[ip]-Temp_k[im]) - Temp_k[i]**2/(Temp_k[ip]-Temp_k[i])**2]) df_ij[i,im]
                #all_Cv_expect[i,2,n] = kB*(A df_ij[ip,i] + B df_ij[i,im]
                A = 2*Temp_k[i]/(Temp_k[ip]-Temp_k[im]) + 4*Temp_k[i]**2/(Temp_k[ip]-Temp_k[im])**2
                B = 2*Temp_k[i]/(Temp_k[ip]-Temp_k[im]) + 4*Temp_k[i]**2/(Temp_k[ip]-Temp_k[im])**2
                #dCv_expect[i,2,n] = kB* [(A ddf_ij[ip,i])**2 + (B sdf_ij[i,im])**2 + 2*A*B*cov(df_ij[ip,i],df_ij[i,im])
                # This isn't it either: need to figure out that last term.
                dCv_expect[i,2] = kB*((A*ddf_ij[ip,i])**2 + (B*ddf_ij[i,im])**2)
                # Would need to add function computing covariance of DDG, (A-B)-(C-D)

        elif (dertype == 'beta'):
            # if beta is evenly spaced, rather than t, we can do 2nd derivative in beta
            # C_v = d<E>/dT = d/dT (df/dbeta) = dbeta/dT d/dbeta (df/dbeta) = -k_b beta^2 df^2/d^2beta
            if (i==originalK) or (i==K-1):
                #Flag as N/A -- we don't try to compute at the endpoints for now
                allCv_expect[i,2,n] = -10000000.0
            else:
                allCv_expect[i,2,n] = kB * beta_k[i]**2 *(df_ij[ip,i]-df_ij[i,im])/((beta_k[ip]-beta_k[im])/(ip-im))**2
            if (n==0):
                dCv_expect[i,2] = kB*(beta_k[i])**2 * (ddf_ij[ip,i]-ddf_ij[i,im])/((beta_k[ip]-beta_k[im])/(ip-im))**2
                # also wrong, need to be fixed.

    if (n==0):
        print('WARNING: only the first derivative (dT) analytic error estimates can currently be trusted.')
        print('They are the only ones reasonably close to bootstrap, within 10-15% at all T.')
        print('')
        PrintResults("Analytic Error Estimates",E_expect,dE_expect,allCv_expect,dCv_expect,types)

if nBoots > 0:
    Cv_boot = np.zeros([K,ntypes],float)
    dCv_boot = np.zeros([K,ntypes],float)
    dE_boot = np.zeros([K])

    for k in range(K):
        for i in range(ntypes):
            # for these averages, don't include the first one, because it's the non-bootstrapped one.
            Cv_boot[k,i] = np.mean(allCv_expect[k,i,1:nBoots_work])
            dCv_boot[k,i] = np.std(allCv_expect[k,i,1:nBoots_work])
            dE_boot[k] = np.std(allE_expect[k,1:nBoots_work])
    PrintResults("Bootstrap Error Estimates",allE_expect[:,0],dE_boot,allCv_expect,dCv_boot,types)


for replica_index_1 in range(len(temperature_list)):
 index = 0
 for replica_index_2 in range(len(temperature_list)):
  E_kn[replica_index_1][index] = replica_energies[replica_index_1][replica_index_2]
  u_kn[replica_index_1][index] = replica_energies[replica_index_1][replica_index_2] / temperature_list[replica_index_2]._value
  index = index + 1
N_k = [len(temperature_list) for i in range(len(temperature_list))]

allCv_expect = np.zeros([len(temperature_list),len(temperature_list)], np.float64)
dCv_expect = np.zeros([len(temperature_list),len(temperature_list)],np.float64)
allE_expect = np.zeros([len(temperature_list)], np.float64)
allE2_expect = np.zeros([len(temperature_list)], np.float64)
dE_expect = np.zeros([len(temperature_list)],np.float64)

mbar = pymbar.MBAR(u_kn,N_k)
beta_k = 1 / (kB * np.array([temperature._value for temperature in temperature_list]))
E_kn = u_kn  # not a copy, we are going to write over it, but we don't need it any more.
for k in range(len(temperature_list)):
  E_kn[:,k]*=beta_k[k]**(-1)
results = mbar.computeExpectations(E_kn, state_dependent = True)
results = {'mu': results[0],'sigma': results[1]}
E_expect = results['mu']
dE_expect = results['sigma']
allE_expect[:] = E_expect[:]

# expectations for the differences, which we need for numerical derivatives
results = mbar.computeExpectations(E_kn,output='differences', state_dependent = True)
results = {'mu': results[0],'sigma': results[1]}
DeltaE_expect = results['mu']
dDeltaE_expect = results['sigma']

results = mbar.computeExpectations(E_kn**2, state_dependent = True)
results = {'mu': results[0],'sigma': results[1]}
E2_expect = results['mu']
dE2_expect = results['sigma']
allE2_expect[:] = E2_expect[:]

results = mbar.getFreeEnergyDifferences()
results = {'Delta_f': results[0],'dDelta_f': results[1]}
df_ij = results['Delta_f']
ddf_ij = results['dDelta_f']

#------------------------------------------------------------------------
# Compute Cv for NVT simulations as <E^2> - <E>^2 / (RT^2)
#------------------------------------------------------------------------

allCv_expect[:,0] = (E2_expect - (E_expect*E_expect)) / ( kB * np.array([temperature._value for temperature in temperature_list])**2)
print(allCv_expect)

exit()
