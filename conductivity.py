# importing packages
import numpy as np
import pandas as pd
import math
from scipy.optimize import bisect


# Shedlovsky model for predicting the equivalent conductivity of single salt solutions
def _equivalent_conductivity(
    conc, temp, epsilon, eta, lambda_0, a,
    z_1, z_2, lambda_0_cation, lambda_0_anion
):
    """
    Calculates the equivalent conductivity of single salt solutions

    Parameters
    ----------
    conc: pandas.Series or list or numpy.ndarray
        Salt concentrations in M
    temp: int or float
        Temperature of the solution in K
    epsilon: int or float
        Dielectric constant of the solvent
    eta: int or float
        Viscosity of the solvent in poise
    lambda_0: int or float
        Limiting equivalent conductivity of the salt in cm^2.S/equiv
    a: int or float
        Distance of closest approach of ions in cm
    z_1: int
        Valency of the cation
    z_2: int
        Valency of the anion
    lambda_0_cation: int or float
        Limiting equivalent conductivity of the cation in cm^2.S/equiv
    lambda_0_anion: int or float
        Limiting equivalent conductivity of the anion in cm^2.S/equiv

    Returns
    -------
    equiv_cond: list
        Equivalent conductivity in cm^2.S/equiv at various salt concentrations
    """
    # calculate the B and q constants
    B = 50.29 * (10 ** 8) * (epsilon * temp) ** (-0.5)
    q = (np.abs(z_1 * z_2) * (lambda_0_cation + lambda_0_anion) /
         ((np.abs(z_1) + np.abs(z_2)) *
          (np.abs(z_2) * lambda_0_cation + np.abs(z_1) * lambda_0_anion)))

    # compute the relaxation (B1) and the electrophoretic (B2) terms
    B_1 = (2.801 * 10 ** 6 * np.abs(z_1 * z_2) * q /
           ((epsilon * temp) ** (3 / 2) * (1 + math.sqrt(q))))
    B_2 = (41.25 * (np.abs(z_1) + np.abs(z_2)) /
           (eta * (epsilon * temp) ** (1 / 2)))

    # calculate the concentration of ionic species
    cation_conc = conc
    Cl_conc = [np.abs(z_1) * conc[i] for i in range(len(conc))]

    # calculate the ionic strength
    I = np.zeros(len(conc))
    for i in range(len(conc)):
        conc_all = [cation_conc[i], Cl_conc[i]]
        z = [z_1, z_2]
        I[i] = 1 / 2 * (sum(conc_all[j] * z[j] ** 2 for j in range(len(conc_all))))

    # calculate the equivalent conductivity of the salt solution
    equiv_cond = []
    for i in range(len(conc)):
        equiv_cond.append(
            lambda_0 - ((B_1 * lambda_0 + B_2) * math.sqrt(I[i]) /
                        (1 + a * B * math.sqrt(I[i])))
        )

    return equiv_cond


# Shedlovsky model for predicting the conductivity of single salt solutions
def shedlovsky(
    conc, temp, epsilon, eta, lambda_0, a,
    z_1, z_2, lambda_0_cation, lambda_0_anion
):
    """
    Calculates the conductivity of single salt solutions

    Parameters
    ----------
    conc: pandas.Series or list or numpy.ndarray
        Salt concentrations in M
    temp: int or float
        Temperature of the solution in K
    epsilon: int or float
        Dielectric constant of the solvent
    eta: int or float
        Viscosity of the solvent in poise
    lambda_0: int or float
        Limiting equivalent conductivity of the salt in cm^2.S/equiv
    a: int or float
        Distance of closest approach of ions in cm
    z_1: int
        Valency of the cation
    z_2: int
        Valency of the anion
    lambda_0_cation: int or float
        Limiting equivalent conductivity of the cation in cm^2.S/equiv
    lambda_0_anion: int or float
        Limiting equivalent conductivity of the anion in cm^2.S/equiv

    Returns
    -------
    specific_cond_calc: list
        Conductivity in milli.S/cm at various salt concentrations
    """

    # check if the arguments are supplied correctly
    if not isinstance(conc, (list, pd.Series, np.ndarray)):
        raise TypeError("Expected a list, pandas.Series, or numpy.ndarray "
                        "for the conc argument.")

    if not isinstance(z_1, int):
        raise TypeError("Expected an integer for the z_1 argument.")

    if not isinstance(z_2, int):
        raise TypeError("Expected an integer for the z_2 argument.")

    if not isinstance(a, (float, int)):
        raise TypeError("Expected a float or integer for the `a` argument.")

    if not isinstance(temp, (float, int)):
        raise TypeError("Expected a float or integer for the temp argument.")

    if not isinstance(eta, (float, int)):
        raise TypeError("Expected a float or integer for the eta argument.")

    if not isinstance(epsilon, (float, int)):
        raise TypeError("Expected a float or integer for the epsilon argument.")

    if not isinstance(lambda_0, (float, int)):
        raise TypeError("Expected a float or integer for the lambda_0 argument.")

    if not isinstance(lambda_0_cation, (float, int)):
        raise TypeError("Expected a float or integer for the lambda_0_cation argument.")

    if not isinstance(lambda_0_anion, (float, int)):
        raise TypeError("Expected a float or integer for the lambda_0_anion argument.")

    if z_1 < 0:
        raise ValueError("The value of z_1 must be a positive integer.")

    if z_2 > 0:
        raise ValueError("The value of z_2 must be a negative integer.")

    # calculate the equivalent conductivity of the salt solution
    equiv_cond = _equivalent_conductivity(
        conc, temp, epsilon, eta, lambda_0, a,
        z_1, z_2, lambda_0_cation, lambda_0_anion
    )

    # calculate the cation concentration
    cation_conc = conc

    # convert the cation concentrations from M (mol/L) to N (equiv/L)
    equiv_conc = [cation_conc[i] * np.abs(z_1) for i in range(len(conc))]

    # calculate the conductivity of the salt solution in S/cm
    specific_cond = []
    for i in range(len(conc)):
        specific_cond.append((equiv_conc[i] / 1000) * equiv_cond[i])

    # convert the conductivity to milli.S/cm
    specific_cond_calc = [specific_cond[i] * 10 ** 3 for i in range(len(conc))]

    return specific_cond_calc


# Mean spherical approximation (MSA) model for multi-salt conductivity predictions
def msa(
    valency, diameters, diff_coeff, temp, eta, epsilon, lambda_0,
    salt_1_conc, salt_2_conc=None, salt_3_conc=None, salt_4_conc=None,
    acidic=False,
):
    """Calculates the conductivity of single, binary, ternary,
     and quaternary salt solutions.

     Note: This function can be extended to solutions containing more
     than four dissolved salts. The current implementation is limited
     to only four salts. It is also limited to salt solutions containing
     a common/shared monovalent anion, e.g., chloride ions

     Note: If the solution is acidic (i.e., one of the salts contain
     hydrogen ion), the concentration of the salt containing the
     hydrogen ion should be supplied last. Additionally, the hydrogen ion
     valency, diameter, diffusion coefficient, and limiting molar
     conductivity should be placed in the last entry after arranging the
     values of other ions in decreasing order of valency

    Parameters
    ----------
    valency: list
        Valency of ions in decreasing order (i.e, highest-valence
        to lowest-valence ions)
    diameters: list
        Hard sphere diameter of the corresponding ions in m
    diff_coeff: list
        Diffusion coefficient of the corresponding ions at
        infinite dilution in m^2/s
    temp: int or float
        Temperature of the solutions in K
    eta: int or float
        Viscosity of the solvent in Pa.s
    epsilon: int or float
        Relative permittivity of the solvent
    lambda_0: list
        Limiting molar conductivity of the corresponding ions in S.m^2/mol
    salt_1_conc: pandas.Series or list or numpy.ndarray
        Concentrations of salt 1 (highest-valence salt) in mM
    salt_2_conc: pandas.Series or list or numpy.ndarray
        Concentrations of salt 2 (second highest-valence salt) in mM
        Optional. Default is None
    salt_3_conc: pandas.Series or list or numpy.ndarray
        Concentrations of salt 3 (third highest-valence salt) in mM
        Optional. Default is None
    salt_4_conc: pandas.Series or list or numpy.ndarray
        Concentrations of salt 4 (lowest-valence salt) in mM,
        Optional. Default is None
    acidic: Boolean
        Specifies if the solution of interest is acidic.
        Optional. Default is False

    Returns
    -------
    cond_calc_con: list
        Conductivity in milli.S/cm at various salt concentrations"""

    # check if the arguments are supplied correctly
    if not isinstance(valency, list):
        raise TypeError("Expected a list for the valency argument.")

    if not isinstance(diameters, list):
        raise TypeError("Expected a list for the diameters argument.")

    if not isinstance(diff_coeff, list):
        raise TypeError("Expected a list for the diff_coeff argument.")

    if not isinstance(temp, (float, int)):
        raise TypeError("Expected a float or integer for the temp argument.")

    if not isinstance(eta, (float, int)):
        raise TypeError("Expected a float or integer for the eta argument.")

    if not isinstance(epsilon, (float, int)):
        raise TypeError("Expected a float or integer for the epsilon argument.")

    if not isinstance(lambda_0, list):
        raise TypeError("Expected a list for the lambda_0 argument.")

    if not isinstance(salt_1_conc, (list, pd.Series, np.ndarray)):
        raise TypeError(
            "salt_1_conc must be a list, pandas.Series, or numpy.ndarray."
        )

    if not isinstance(acidic, bool):
        raise TypeError("Expected a Boolean for the `acidic` argument.")

    # constants
    Avogadros_num = 6.022 * 10 ** 23  # Avogadro's constant
    k_B = 1.381 * 10 ** (-23)  # Boltzmann constant in J/K
    charge = 1.602 * 10 ** (-19)  # elementary charge in C
    epsilon_0 = 8.854 * 10 ** (-12)  # permittivity of free space in F/m
    Faraday = 96500  # Faraday's constant in C/mol

    # number of data points
    ndata = len(salt_1_conc)
    if ndata == 0:
        raise ValueError("salt_1_conc cannot be empty.")

    # number of ionic species
    n_species = len(valency)

    # charge of ionic species
    ion_charge = [valency[i] * charge for i in range(len(valency))]

    # combine the first salt with all additional salts.
    all_salt_conc = [salt_1_conc, salt_2_conc, salt_3_conc, salt_4_conc]

    for i, salt_conc in enumerate(all_salt_conc):
        if i == 0:
            pass
        else:
            if salt_conc is not None and len(salt_conc) != ndata:
                raise ValueError(
                    "All the salt concentrations must have the same length."
                )
            elif (salt_conc is not None
                  and not isinstance(salt_conc, (list, pd.Series, np.ndarray))):
                raise TypeError(
                    "All salt concentrations must be a list, pandas.Series, "
                    "or numpy.ndarray."
                )

    # retain only the concentrations that were supplied and convert them
    # to floating-point NumPy arrays
    salt_conc_supplied = [
        np.asarray(salt_conc, dtype=float) for salt_conc in all_salt_conc
        if salt_conc is not None
    ]

    # get the number of active salts
    n_salts = len(salt_conc_supplied)

    # convert salt concentrations to number densities
    number_density_salts = [
        salt_conc * Avogadros_num for salt_conc in salt_conc_supplied
    ]

    # cation number densities are the same as the corresponding salt densities
    number_density_cats = number_density_salts

    # anion number density is the sum of cation valency times salt density
    if not acidic:
        number_density_an = np.sum(
            [valency[i] * number_density_salts[i] for i in range(n_salts)],
            axis=0,
        )
    else:
        # the ordering of the ions is:
        # [cation_1, cation_2, ..., anion, H+]
        anion_valency = valency[-2]
        hydrogen_valency = valency[-1]

        # calculate the number density of all the non-H+ salts
        positive_number_density = np.sum(
            [valency[i] * number_density_salts[i] for i in range(n_salts - 1)],
            axis=0,
        )

        # the H+ salt is the final supplied salt
        # update the number density calculation
        positive_number_density += hydrogen_valency * number_density_salts[-1]

        number_density_an = positive_number_density

    # calculate the bulk conductivity of the salt solution
    all_cond_calc = np.zeros(ndata)
    for n in range(ndata):  # loop through the data points
        if not acidic:
            number_density = [
                number_density_cat[n] for number_density_cat in number_density_cats
            ]
            number_density.append(number_density_an[n])
        else:
            # HCl is always the last supplied salt, so the last cation
            # number density corresponds to H+
            hydrogen_number_density = number_density_cats[-1][n]

            # add the number density of all cations except H+
            number_density = [
                number_density_cat[n]
                for number_density_cat in number_density_cats[:-1]
            ]

            # order the number density as follows:
            # [other cations, common anion, H+]
            number_density.append(number_density_an[n])
            number_density.append(hydrogen_number_density)

        # calculate the ionic mobility (omega) and relative ionic strength (mew) of all species
        omega = np.zeros(n_species)
        mew = np.zeros(n_species)
        for a in range(n_species):
            omega[a] = diff_coeff[a] / (k_B * temp)
            mew[a] = (number_density[a] * ion_charge[a] ** 2 /
                      sum(number_density[j] * ion_charge[j] ** 2 for j in range(n_species)))

        # calculate the Debye length
        kappa = math.sqrt(
            sum(number_density[l] * ion_charge[l] ** 2 /
                (epsilon * epsilon_0 * k_B * temp) for l in range(n_species)))

        # calculate the average mobility of all species
        omega_bar = sum(mew[j] * omega[j] for j in range(n_species))

        # calculate the transport number of all species
        transport_num = np.zeros(n_species)
        for j in range(n_species):
            transport_num[j] = mew[j] * omega[j] / omega_bar

        # calculate the value of alpha from the equation containing alpha
        alpha_scaled = np.zeros(n_species)
        def func_alpha(alpha_scaled):
            """
            Evaluates residuals of the equation containing alpha

            Arguments:
                alpha_scaled = alpha/omega_bar

            Returns:
                residual of the equation containing alpha
            """
            return ((alpha_scaled) * sum(
                transport_num[k] /
                ((omega[k] / omega_bar) ** 2 - (alpha_scaled) ** 2)
                for k in range(n_species)))

        # since alpha is scaled by omega_bar, omega need to be scaled by same factor
        omega_scaled = np.zeros(n_species)
        for k in range(n_species):
            omega_scaled[k] = omega[k] / omega_bar

        # updating the values of alpha_scaled
        for p in range(n_species):
            if p == 0:
                alpha_scaled[p] = bisect(func_alpha, 0, 0.991 * omega_scaled[p])
            else:
                try:
                    alpha_scaled[p] = bisect(func_alpha, 1.0001 * omega_scaled[p - 1],
                                             0.9991 * omega_scaled[p])
                except ValueError:
                    raise ValueError(
                        "The current order of ions which is defined by the valency "
                        "argument (propagated to other arguments as well) violates "
                        "the mobility constraints. If hydrogen ion is present in the "
                        "solution, place its valency, diameter, diffusion coefficient, "
                        "and limiting molar conductivity in the last entry after "
                        "arranging the values of other ions in decreasing order of "
                        "valency."
                    )

        # re-scaling alpha_scaled back to alpha
        alphas = [alpha_scaled[p] * omega_bar for p in range(n_species)]

        # calculate the q and N values of all species
        q = np.zeros(n_species)
        N = np.zeros(n_species)
        for p in range(n_species):
            q[p] = sum(omega_bar * transport_num[r] /
                       (omega[r] + alphas[p]) for r in range(n_species))
            N[p] = math.sqrt(1 / sum(
                transport_num[r] * omega[r] ** 2 /
                (omega[r] ** 2 - alphas[p] ** 2) ** 2 for r in range(n_species)))

        # compute the zeta matrix
        zeta = np.zeros((n_species, n_species))

        # calculate the electrophoretic correction
        delta_vel_divide_vel = np.zeros(n_species)
        for i in range(n_species):
            # update the electrophoretic correction
            prefactor = (-(Faraday ** 2) * np.abs(valency[i]) /
                         (12 * math.pi * epsilon_0 * epsilon * eta *
                          k_B * temp * Avogadros_num * lambda_0[i]))
            delta_vel_divide_vel[i] = (
                    prefactor * sum(number_density[j] * ion_charge[j] ** 2 *
                                    ((np.exp(kappa * (diameters[i] - 0.5 * (diameters[i] + diameters[j])))
                                      / (kappa * (1 + kappa * diameters[i]))) +
                                     (np.exp(kappa * (diameters[j] - 0.5 * (diameters[i] + diameters[j])))
                                      / (kappa * (1 + kappa * diameters[j])))) for j in range(n_species)))

            # update the zeta matrix
            for p in range(n_species):
                zeta[i][p] = N[p] * omega[i] / (omega[i] ** 2 - alphas[p] ** 2)

        # calculate the conductivity of all ionic species
        cond_species = np.zeros(n_species)

        # calculate the relaxation correction
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
                        term1 = (transport_num[j] * zeta[j][p] * mew[a] *
                                 (ion_charge[a] * omega[a] - ion_charge[j] * omega[j]) /
                                 (ion_charge[a] * ion_charge[j] * (omega[a] + omega[j])))

                        # second fraction of summation term
                        term2 = (math.sinh(kappa * math.sqrt(q[p]) * sigma_aj) /
                                 (kappa * math.sqrt(q[p]) * sigma_aj))

                        # evaluate the integral term
                        integral_prefactor = (- (charge ** 2 * valency[a] * valency[j]) /
                                              (8 * math.pi * epsilon_0 * epsilon * k_B * temp))

                        integral_term1 = np.exp(kappa * diameters[j]) / (1 + kappa * diameters[j])
                        integral_term2 = np.exp(kappa * diameters[a]) / (1 + kappa * diameters[a])

                        integral_exp_term = (np.exp(-kappa * (1 + np.sqrt(q[p])) * sigma_aj) /
                                             (kappa * (1 + np.sqrt(q[p]))))

                        integral = (integral_prefactor * (integral_term1 + integral_term2) *
                                    integral_exp_term)

                        inner_delta_k_divide_k_sum += term1 * term2 * integral

                delta_k_divide_k_sum += zeta[i][p] * inner_delta_k_divide_k_sum

            # update the relaxation correction
            delta_k_divide_k[i] = -(kappa ** 2) * ion_charge[i] * delta_k_divide_k_sum / 3

            # update the conductivity of all ionic species
            cond_species[i] = ((charge ** 2) * number_density[i] * diff_coeff[i] *
                               (valency[i] ** 2) * (1 + delta_vel_divide_vel[i]) *
                               (1 + delta_k_divide_k[i]) / (k_B * temp))

        # compute the bulk conductivity of the salt solution in S/m
        all_cond_calc[n] = sum(cond_species)

    # convert the conductivity to milli.S/cm
    cond_calc_con = [all_cond_calc[i] * 10 ** 3 / 100 for i in range(ndata)]

    print(f"Relaxation correction: {delta_k_divide_k}")
    print(f"Hydrodynamic correction: {delta_vel_divide_vel}")
    print(f"Relative ionic strength: {mew}")
    print(f"Transport number: {transport_num}")
    print(f"Ionic mobility: {omega}")
    print(f"Average mobility: {omega_bar}")
    print(f"Alpha: {alphas}")
    print(f"zeta matrix: {zeta}")

    return cond_calc_con
