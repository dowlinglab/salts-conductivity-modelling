# importing packages
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.stats as stats
import pandas as pd
import pyomo.environ as pyo

# Shedlovsky model for electrolyte equivalent conductivity
def _shedlovsky(conc, temp, epsilon, eta, lambda_0, a, z_1, z_2, lambda_0_cation, lambda_0_anion):
    """
    Calculates equivalent conductivity using equation (7.36) in reference

    Arguments:
        conc: integer or float value of the salt concentration in M
        temp: integer or float value of the temperature in K
        epsilon:integer or float value of the dielectric constant
        eta: integer or float value of the viscosity of the salt solution in poise
        lambda_0: integer or float value of the limiting equivalent conductivity of
                the salt in cm^2.S/equiv
        a: integer or float value of the distance within which no other ions can penetrate in cm
        z_1: integer value of the valency of the cation
        z_2: integer value of the valency of the anion
        lambda_0_cation: integer or float value of the limiting equivalent conductivity of
                        the cation in cm^2.S/equiv
        lambda_0_anion: integer or float value of the limiting equivalent conductivity of
                        the anion in cm^2.S/equiv

    Returns:
        equiv_cond: equivalent conductivity of the salt in cm^2.S/equiv
    """
    # calculating B and q
    B = 50.29 * (10 ** 8) * (epsilon * temp) ** (-0.5)
    q = np.abs(z_1 * z_2) * (lambda_0_cation + lambda_0_anion) / (
                (np.abs(z_1) + np.abs(z_2)) * (np.abs(z_2) * lambda_0_cation + np.abs(z_1) * lambda_0_anion))

    # B1 and B2
    B_1 = 2.801 * 10 ** 6 * np.abs(z_1 * z_2) * q / ((epsilon * temp) ** (3 / 2) * (1 + pyo.sqrt(q)))
    B_2 = 41.25 * (np.abs(z_1) + np.abs(z_2)) / (eta * (epsilon * temp) ** (1 / 2))

    # ion concentration
    cation_conc = conc
    Cl_conc = np.abs(z_1) * conc

    # calculate the ionic strength
    conc_all = [cation_conc, Cl_conc]
    z = [z_1, z_2]
    I = 1 / 2 * (sum(conc_all[j] * z[j] ** 2 for j in range(len(conc_all))))

    # calculate the salt equivalent conductivity
    equiv_cond = lambda_0 - ((B_1 * lambda_0 + B_2) * pyo.sqrt(I) / (1 + a * B * pyo.sqrt(I)))

    return equiv_cond


# variant Shedlovsky model for predicting electrolyte specific conductivity
def variant_shedlovsky(conc, temp, epsilon, eta, lambda_0, a, z_1, z_2, lambda_0_cation, lambda_0_anion):
    """
    Calculates specific conductivity from equivalent conductivity

    Arguments:
        conc: integer or float value of the salt concentration in M
        temp: integer or float value of the temperature in K
        epsilon:integer or float value of the dielectric constant
        eta: integer or float value of the viscosity of the salt solution in poise
        lambda_0: integer or float value of the limiting equivalent conductivity of
                the salt in cm^2.S/equiv
        a: integer or float value of the distance within which no other ions can penetrate in cm
        z_1: integer value of the valency of the cation
        z_2: integer value of the valency of the anion
        lambda_0_cation: integer or float value of the limiting equivalent conductivity of
                        the cation in cm^2.S/equiv
        lambda_0_anion: integer or float value of the limiting equivalent conductivity of
                        the anion in cm^2.S/equiv

    Returns:
        specific_cond_calc: specific conductivity of the salt in micro.S/cm
    """

    # calculate the salt equivalent conductivity
    equiv_cond = _shedlovsky(conc, temp, epsilon, eta, lambda_0, a, z_1, z_2, lambda_0_cation, lambda_0_anion)

    # cation concentration
    cation_conc = conc

    # convert the cation concentrations from M to equivalent per litre
    equiv_conc = cation_conc * np.abs(z_1)

    # calculate the salt specific conductivity in Siemen per cm
    specific_cond = (equiv_conc / 1000) * equiv_cond

    # specific conductivity in micro Siemen per cm
    specific_cond_calc = specific_cond * 10 ** 6

    return specific_cond_calc