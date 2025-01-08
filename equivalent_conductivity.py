# importing packages
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.stats as stats
import pandas as pd

# equivalent conductivity model
def cond_calc_ref(conc, temp, epsilon, eta, lambda_0, a, z_1, z_2, lambda_0_cation, lambda_0_anion):
    """
    Calculates equivalent conductivity using equation (7.36) in reference

    Arguments:
        conc: salt concentration in M
        temp: temperature of salt solution in K
        epsilon: dielectric constant
        eta: viscosity of salt solution in poise
        lambda_0: limiting equivalent conductivity of salt in cm^2.Siemen.equiv^(-1)
        a: distance within which no other ions can penetrate in cm
        z_1: valency of cation
        z_2: valency of anion
        lambda_0_cation: limiting equivalent conductivity of cation in cm^2.Siemen.equiv^(-1)
        lambda_0_anion: limiting equivalent conductivity of anion in cm^2.Siemen.equiv^(-1)

    Returns:
        equiv_cond_ref: equivalent conductivity values in cm^2.S/equiv
    """
    # calculating B and q
    B = 50.29 * (10 ** 8) * (epsilon * temp) ** (-0.5)
    q = np.abs(z_1 * z_2) * (lambda_0_cation + lambda_0_anion) / (
                (np.abs(z_1) + np.abs(z_2)) * (np.abs(z_2) * lambda_0_cation + np.abs(z_1) * lambda_0_anion))

    # B1 and B2
    B_1 = 2.801 * 10 ** 6 * np.abs(z_1 * z_2) * q / ((epsilon * temp) ** (3 / 2) * (1 + math.sqrt(q)))
    B_2 = 41.25 * (np.abs(z_1) + np.abs(z_2)) / (eta * (epsilon * temp) ** (1 / 2))

    # species concentration
    cation_conc = conc
    Cl_conc = [np.abs(z_1) * conc[i] for i in range(len(conc))]

    # ionic strength
    I = np.zeros(len(conc))
    for i in range(len(conc)):
        conc_all = [cation_conc[i], Cl_conc[i]]
        z = [z_1, z_2]
        I[i] = 1 / 2 * (sum(conc_all[j] * z[j] ** 2 for j in range(len(conc_all))))

    # equivalent conductivity
    equiv_cond_ref = []
    for i in range(len(conc)):
        equiv_cond_ref.append(lambda_0 - ((B_1 * lambda_0 + B_2) * math.sqrt(I[i]) / (1 + a * B * math.sqrt(I[i]))))

    return (equiv_cond_ref)


# function the predicts specific conductivity
def specific_cond_model(conc, temp, epsilon, eta, lambda_0, a, z_1, z_2, lambda_0_cation, lambda_0_anion):
    """
    Calculates specific conductivity from equivalent conductivity

    Arguments:
        conc: salt concentration in M
        temp: temperature of salt solution in K
        epsilon: dielectric constant
        eta: viscosity of salt solution in poise
        lambda_0: limiting equivalent conductivity of salt in cm^2.Siemen.equiv^(-1)
        a: distance within which no other ions can penetrate in cm
        z_1: valency of cation
        z_2: valency of anion
        lambda_0_cation: limiting equivalent conductivity of cation in cm^2.Siemen.equiv^(-1)
        lambda_0_anion: limiting equivalent conductivity of anion in cm^2.Siemen.equiv^(-1)

    Returns:
        specific_cond_calc: specific conductivity values in micro Siemen per cm
    """

    # equivalent conductivity
    equiv_cond = cond_calc_ref(conc, temp, epsilon, eta, lambda_0, a, z_1, z_2, lambda_0_cation, lambda_0_anion)

    # species concentration
    cation_conc = conc
    Cl_conc = [np.abs(z_1) * conc[i] for i in range(len(conc))]

    # converting the concentrations in mM to equivalent per litre
    equiv_conc = [cation_conc[i] * np.abs(z_1) for i in range(len(conc))]

    # specific conductivity in Siemen per cm
    specific_cond = []
    for i in range(len(conc)):
        specific_cond.append((equiv_conc[i] / 1000) * equiv_cond[i])

    # specific conductivity in micro Siemen per cm
    specific_cond_calc = [specific_cond[i] * 10 ** 6 for i in range(len(conc))]

    return specific_cond_calc