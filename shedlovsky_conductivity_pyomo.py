# importing packages
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.stats as stats
from scipy.optimize import fsolve, bisect
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


# Mean spherical approximation (MSA) transport model for multi-salt specific conductivity
def msa_transport(valency, diameters, diff_coeff, T, sensor_voltage, probe_distance, eta, epsilon,
                           salt_1_conc, salt_2_conc=None, salt_3_conc=None):
    """Calculates the specific conductivity of single-salt, two-salt, and three-salt solutions

    Argument:
        valency: list of the valency of ions with cations listed before anions in decreasing order
        diameters: list of the corresponding hard sphere diameter of the ions in m
        diff_coeff: list of the corresponding diffusion coefficient of the ions at infinite dilution in m^2/s
        T: integer or float value of the temperature in K
        sensor_voltage: integer or float value of the sensor measuring voltage in V
        probe_distance: integer or float value of the distance between the probes in m
        eta: integer or float value of the viscosity in Pa.s
        epsilon: integer or float value of the relative permittivity
        salt_1_conc = integer or float value of the concentration of salt 1 in mM
        salt_2_conc = integer or float value of the concentration of salt 2 in mM
        salt_3_conc = integer or float value of the concentration of salt 3 in mM

    Returns:
        all_cond_calc_con: specific conductivity of the salt solution in micro.S/cm"""

    # constants
    Avogadros_num = 6.022 * 10 ** 23 # Avogadro's constant
    k_B = 1.381 * 10 ** (-23) # Boltzmann constant in J/K
    charge = 1.602 * 10 ** (-19) # elementary charge in C
    epsilon_0 = 8.854 * 10 ** (-12) # permittivity of free space in F/m

    # electric field
    E_field = sensor_voltage/probe_distance

    # number of species
    n_species = len(valency)

    # species charge
    ion_charge = [valency[i] * charge for i in range(len(valency))]

    # converting molar concentrations to number density
    if salt_2_conc is None and salt_3_conc is None:
        number_density_salt_1 = salt_1_conc * Avogadros_num

        # valency of cation
        valency_cation = valency[0]

        # evaluating the number density of individual species
        number_density_cat_1 = number_density_salt_1
        number_density_an = valency_cation * number_density_salt_1

        # number density of species
        number_density = [number_density_cat_1, number_density_an]
    elif salt_2_conc is not None and salt_3_conc is None:
        number_density_salt_1 = salt_1_conc * Avogadros_num
        number_density_salt_2 = salt_2_conc * Avogadros_num

        # valency of cations
        valency_cation_1 = valency[0]
        valency_cation_2 = valency[1]

        # evaluating the number density of individual species
        number_density_cat_1 = number_density_salt_1
        number_density_cat_2 = number_density_salt_2
        number_density_an = valency_cation_1 * number_density_salt_1 + valency_cation_2 * number_density_salt_2

        # number density of species
        number_density = [number_density_cat_1, number_density_cat_2, number_density_an]
    else:
        number_density_salt_1 = salt_1_conc * Avogadros_num
        number_density_salt_2 = salt_2_conc * Avogadros_num
        number_density_salt_3 = salt_3_conc * Avogadros_num

        # valency of cations
        valency_cation_1 = valency[0]
        valency_cation_2 = valency[1]
        valency_cation_3 = valency[2]

        # evaluating the number density of individual species
        number_density_cat_1 = number_density_salt_1
        number_density_cat_2 = number_density_salt_2
        number_density_cat_3 = number_density_salt_3
        number_density_an = valency_cation_1 * number_density_salt_1 + valency_cation_2 * number_density_salt_2 + valency_cation_3 * number_density_salt_3

        # number density of species
        number_density = [number_density_cat_1, number_density_cat_2, number_density_cat_3, number_density_an]

    # calculate the species ionic mobility (omega) and the relative ionic strength (mew)
    omega = np.zeros(n_species)
    mew = np.zeros(n_species)
    for a in range(n_species):
        omega[a] = diff_coeff[a] / (k_B * T)
        mew[a] = number_density[a] * ion_charge[a] ** 2 / sum(
            number_density[j] * ion_charge[j] ** 2 for j in range(n_species))

    # calculate the Debye length from equation (5)
    kappa = pyo.sqrt(
        sum(number_density[l] * ion_charge[l] ** 2 / (epsilon * epsilon_0 * k_B * T) for l in range(n_species)))

    # calculate the mean mobility from equation (7)
    omega_bar = sum(mew[j] * omega[j] for j in range(n_species))

    # calculate the transport number of all species from equation (8)
    transport_num = np.zeros(n_species)
    for j in range(n_species):
        transport_num[j] = mew[j] * omega[j] / omega_bar

    # calculate delta from equation (21)
    delta = 1 - ((math.pi / 6) * sum(number_density[k] * diameters[k] ** 3 for k in range(n_species)))

    # function that evaluates equation (18)
    def func_gamma(gamma):
        """
        Evaluates residual of equation (18)

        Arguments:
            gamma

        Returns:
            Residual of the equation
        """
        # calculating capital omega from equation (20)
        capital_omega = 1 + ((math.pi / (2 * delta)) * sum(
            number_density[k] * diameters[k] ** 3 / (1 + gamma * diameters[k]) for k in range(n_species)))

        # calculating Pn from equation (19)
        P_n = (1 / capital_omega) * sum(
            number_density[k] * diameters[k] * valency[k] / (1 + gamma * diameters[k]) for k in range(n_species))

        return (4 * epsilon_0 * gamma ** 2 - ((charge ** 2) / (epsilon * k_B * T)) *
                sum(number_density[l] * ((valency[l] - (math.pi * P_n * diameters[l] ** 2 / (2 * delta))) / (
                            1 + gamma * diameters[l])) ** 2 for l in range(n_species)))

    # finding gamma, the root of equation (18)
    gamma = bisect(func_gamma, 0, 0.5 * kappa)

    # re-evaluating capital_omega, P_n and nu
    capital_omega = 1 + ((math.pi / (2 * delta)) * sum(
        number_density[k] * diameters[k] ** 3 / (1 + gamma * diameters[k]) for k in range(n_species)))
    P_n = (1 / capital_omega) * sum(
        number_density[k] * diameters[k] * valency[k] / (1 + gamma * diameters[k]) for k in range(n_species))

    # calculate the nu values of all species from equation (17)
    nu = np.zeros(n_species)
    for k in range(n_species):
        nu[k] = (gamma * valency[k] / (1 + gamma * diameters[k])) + (math.pi / (2 * delta)) * (
                    P_n * diameters[k] / (1 + gamma * diameters[k]))

    # evaluating q and N for all species from equation (9) and (11) respectively
    q = np.zeros(n_species)
    N = np.zeros(n_species)
    alpha_scaled = np.zeros(n_species)

    # function that evaluates equation (12)
    def func_alpha(alpha_scaled):
        """
        Evaluates residuals of equation 12

        Arguments:
            alpha_scaled = alpha/omega_bar

        Returns:
            residual of equation 12
        """
        return ((alpha_scaled) * sum(
            transport_num[k] / ((omega[k] / omega_bar) ** 2 - (alpha_scaled) ** 2) for k in range(n_species)))

    # since alpha is scaled by omega_bar, omega need to be scaled by same factor
    omega_scaled = np.zeros(n_species)
    for k in range(n_species):
            omega_scaled[k] = omega[k] / omega_bar

    # updating the values of alpha_scaled
    for p in range(n_species):
        if p == 0:
            alpha_scaled[p] = bisect(func_alpha, 0, 0.991 * omega_scaled[p])
        else:
            alpha_scaled[p] = bisect(func_alpha, 1.001 * omega_scaled[p-1], 0.991 * omega_scaled[p])

    # re-scaling alpha_scaled back to alpha
    alphas = [alpha_scaled[p] * omega_bar for p in range(n_species)]

    # updating q and N values
    for p in range(n_species):
        q[p] = sum(omega_bar * transport_num[r] / (omega[r] + alphas[p]) for r in range(n_species))
        N[p] = pyo.sqrt(1 / sum(
            transport_num[r] * omega[r] ** 2 / (omega[r] ** 2 - alphas[p] ** 2) ** 2 for r in range(n_species)))

    # ksi matrix
    ksi = np.zeros((n_species, n_species))

    # calculate the velocity of the species
    vel = np.zeros(n_species)

    # calculate the change in the velocity of the species
    delta_vel = np.zeros(n_species)

    # calculate the hydrodynamic correction
    delta_vel_divide_vel = np.zeros(n_species)
    for i in range(n_species):
        # update the velocity of species from equation (2)
        vel[i] = valency[i] * charge * E_field * diff_coeff[i] / (k_B * T)

        # update the change in the velocity of species from equation (16)
        delta_vel[i] = -(charge * E_field / (3 * math.pi * eta)) * (nu[i] + (math.pi / 4) * sum(
            number_density[j] * valency[j] * diameters[j] ** 2 for j in range(n_species))
                                                                    - (math.pi / 6) * sum(
                    number_density[j] * (diameters[j] ** 3) * nu[j] for j in range(n_species)))

        # update the hydrodynamic correction; equation (16) divide equation (2)
        delta_vel_divide_vel[i] = delta_vel[i] / vel[i]

        # update the ksi matrix
        for p in range(n_species):
            ksi[i][p] = N[p] * omega[i] / (omega[i] ** 2 - alphas[p] ** 2)

    # conductivity of species from equation (1)
    cond_species = np.zeros(n_species)

    # calculate the relaxation correction from equation (4)
    delta_k_divide_k = np.zeros(n_species)
    for i in range(n_species):
        delta_k_divide_k_sum = 0.0
        for p in range(n_species):
            inner_delta_k_divide_k_sum = 0.0
            for j in range(n_species):
                for a in range(n_species):
                    # calculate the average diameter
                    sigma_aj = 0.5 * (diameters[a] + diameters[j])

                    # first fraction of summation term
                    term1 = transport_num[j] * ksi[j][p] * mew[a] * (
                                ion_charge[a] * omega[a] - ion_charge[j] * omega[j]) / (
                                        ion_charge[a] * ion_charge[j] * (omega[a] + omega[j]))

                    # second fraction of summation term
                    term2 = math.sinh(kappa * pyo.sqrt(q[p]) * sigma_aj) / (kappa * pyo.sqrt(q[p]) * sigma_aj)

                    # integral numerator from equation (14)
                    integral_num = -ion_charge[a] * ion_charge[j] * kappa * pyo.sqrt(q[p]) * sigma_aj * np.exp(
                        -kappa * pyo.sqrt(q[p]) * sigma_aj)

                    # calculate Y from equation (15)
                    Y_num = sum(
                        number_density[m] * (valency[m] ** 2) * pyo.exp(-kappa * math.sqrt(q[p]) * diameters[m]) / (
                                    1 + gamma * diameters[m]) ** 2 for m in range(n_species))
                    Y_denom = sum(number_density[m] * valency[m] ** 2 / (1 + gamma * diameters[m]) ** 2 for m in
                                  range(n_species))
                    Y = Y_num / Y_denom

                    # integral denominator from equation (14)
                    integral_denom = 4 * math.pi * epsilon_0 * epsilon * k_B * T * (
                                (kappa ** 2) * q[p] + 2 * gamma * kappa * pyo.sqrt(q[p]) + 2 * gamma ** 2 - 2 * (
                                    gamma ** 2) * Y)

                    inner_delta_k_divide_k_sum += term1 * term2 * (integral_num / integral_denom)

            delta_k_divide_k_sum += ksi[i][p] * inner_delta_k_divide_k_sum

        # relaxation correction from equation (4)
        delta_k_divide_k[i] = -(kappa ** 2) * ion_charge[i] * delta_k_divide_k_sum / 3

        # calculate the conductivity of the species from equation (1)
        cond_species[i] = (charge ** 2) * number_density[i] * diff_coeff[i] * (valency[i] ** 2) * (
                    1 + delta_vel_divide_vel[i]) * (1 + delta_k_divide_k[i]) / (k_B * T)

    # calculate the bulk conductivity from equation (1)
    all_cond_calc = sum(cond_species)

    # convert the calculated conductivity from S/m to micro.S/cm
    cond_calc_con = all_cond_calc * 10 ** 6 / 100

    return cond_calc_con