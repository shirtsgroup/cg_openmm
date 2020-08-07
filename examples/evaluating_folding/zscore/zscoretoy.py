import numpy as np
import matplotlib.pyplot as plt

e_unfolded_mean = 0  # unfolded mean energy
e_unfolded_sigma = 3  # unfolded standard deviation energy
n_unfolded = 1000  # ratio of number of unfolded states to folded states. We keep number of folded states at 1
e_folded = -15  # energy of the unfolded state
Tmax = 10  # temperature max to plot; T in units of kBT
Zscore = (e_folded - e_unfolded_mean) / (e_unfolded_sigma)  # Z-score for this folded state

# We will generate data from the specified unfolded Gaussian
# distibution; if we do the calcuation analytically, then we will, at
# low T, always find some structure that is lower than the folded
# state (Gaussian has population everywhere), so the average energy
# will go to zero as T->0, the folded state won't be the lowest state,
# etc.

# generate the unfolded state energies
unfolded = (e_unfolded_sigma * np.random.normal(size=n_unfolded)) - e_unfolded_mean

# stack all the energies together
all = np.concatenate([np.array(e_folded), unfolded], axis=None)

# now compute thermodynamic properties as a function of T
toplotT = np.linspace(0.1, Tmax, 500)
Tlen = len(toplotT)
A = np.zeros(Tlen)
E = np.zeros(Tlen)
S = np.zeros(Tlen)
C = np.zeros(Tlen)
Pf = np.zeros(Tlen)

for i, T in enumerate(toplotT):
    Q = np.sum(np.exp(-all / T))  # partition function
    A[i] = -T * np.log(Q)  # free energy

    # plot <E> vs. T
    # <E> = kT \ln \int E omega(E) exp(-E/T) / Q.

    E[i] = np.sum(all * np.exp(-all / T))
    E[i] = E[i] / Q

    # plot S vs. T
    # A = E - TS
    # S = (E - A)/T
    S[i] = (E[i] - A[i]) / T

    # plot C_V vs T
    # C_V = (<E^2>-<E>)^T^2 =
    # E^2 =  \int E^2 omega(E) exp(-E/T) / Q
    E2 = np.sum(all ** 2 * np.exp(-all / T))
    E2 = E2 / Q
    C[i] = (E2 - E[i] ** 2) / T ** 2

    # percent folded
    folded = np.exp(-e_folded / T)
    Pf[i] = folded / Q

toplot = [A, Pf, E, S, C]
titles = [
    "Helmholtz free energy (A) vs. T",
    "Percent folded vs. T",
    "Energy (E) vs. T",
    "Entropy (S) vs. T",
    "Heat Capacity (C) vs. T",
]
ylabels = ["A", "% folded", "E", "S", "C"]

file_names = ["A_v_T", "Percent_folded_v_T", "E_v_T", "S_v_T", "C_v_T"]

for p, t, y, file_name in zip(toplot, titles, ylabels, file_names):
    plt.plot(toplotT, p)
    plt.title(t)
    plt.xlabel("T")
    plt.ylabel(y)
    plt.show()
    plt.savefig(str(str(file_name) + ".png"))
