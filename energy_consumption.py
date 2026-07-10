import numpy as np
import math
from cycle_times import *
import matplotlib.pyplot as plt
import sys


def Davis_eq(coeffs, params, v, N, Np):
    A = coeff_A(coeffs, params, N, Np)
    B = coeff_B(coeffs, params, N, Np)
    C = coeff_C(coeffs, params, N)
    r = A + B * v + C * math.pow(v, 2)
    return (r, A, B, C)

def coeff_A(coeffs, params, N, Np):
    A = coeffs['a1'] * (N - Np) * params['mass'] + coeffs['a2'] * (Np) * params['mass']
    return A

def coeff_B(coeffs, params, N, Np): #, N_p=2, mass=773.8 / N):
    B = coeffs['b1'] * N * params['mass'] + coeffs['b2'] * (N - Np) + coeffs['b3'] * Np * params['P']
    return B

def coeff_C(coeffs, params, N): #, N_p=2, C_dn=0.15, C_db=0.6, area=area, Pe=Pe, l_car=l_car, l_gap=l_gap, N_b=2, N_pt=2):
    C = coeffs['c1'] * params['C_dn'] * params['area'] + \
        coeffs['c2'] * params['Pe'] * params['l_car'] * N + \
        coeffs['c3'] * params['Pe'] * params['l_gap'] * (N - 1) + \
        coeffs['c4'] * params['C_db'] * (N-1) + \
        coeffs['c5'] * params['N_pt']
    return C


def E_cc(coeffs, params, v, N, Z=None, mode='n_station', Np=2):
    energy = Davis_eq(coeffs, params, v, N, Np)[0] * v * C_cc(params, v, mode=mode)
    return energy / 3600 / 1000


def E_ca(coeffs, params, v, N, p=None, Z=None, mode='n_station', Np=2):
    energy = 1000 * params['mass'] * N * math.pow(v, 2) + 2 / params['a'] * (
                1 / 2 * coeff_A(coeffs, params, N, Np) * math.pow(v, 2) + \
                1 / 3 * coeff_B(coeffs, params, N, Np) * math.pow(v, 3) + \
                1 / 4 + coeff_C(coeffs, params, N) * math.pow(v, 4))

    return energy / 3600 / 1000


def E_cd(coeffs, params, v, N, Z=None, mode='n_station', Np=2):
    tmp_a = coeff_C(coeffs, params, N)
    tmp_b = coeff_B(coeffs, params, N, Np)
    tmp_c = coeff_A(coeffs, params, N, Np) - 1000 * params['mass'] * N * params['a']

    v_ = (-tmp_b + math.sqrt(tmp_b ** 2 - 4 * tmp_c * tmp_a)) / (2 * tmp_a)
    v_ = np.min([300 / 3.6, v_])

    energy = params['eta_reg'] * (1000 * params['mass'] * N * math.pow(v_, 2) -
                                  2 / params['a'] * (1 / 2 * coeff_A(coeffs, params, N, Np) * math.pow(v_, 2) + \
                                                     1 / 3 * coeff_B(coeffs, params, N, Np) * math.pow(v_, 3) - \
                                                     1 / 4 * coeff_C(coeffs, params, N) * math.pow(v_, 4)))

    return energy / 3600 / 1000


def E_dc(coeffs, params, v, N, Z=None, mode='n_station', Np=2):
    if mode == 'n_station':
        if Z is not None:
            energy = 0
            for z in Z:
                energy += Davis_eq(coeffs, params, v, N - z, Np)[0] * v * C_dc(params, v, mode=mode) / (
                            2 * (params['n_station'] - 2))
        else:
            raise TypeError("Z should have value for n_station mode")
    else:
        energy = Davis_eq(coeffs, params, v, N - params['N_dc'], Np)[0] * v * C_dc(params, v, mode=mode)
    return energy / 3600 / 1000


def E_da(coeffs, params, v, N, Z=None, mode='n_station', Np=1):
    if mode == 'n_station':
        if Z is not None:
            energy = 0
            for z in Z:
                e_ = 1 / 2 * 1000 * params['mass'] * z * math.pow(v, 2) + 1 / params['a'] * (
                        1 / 2 * coeff_A(coeffs, params, params['N_dc'], Np) * math.pow(v, 2) + \
                        1 / 3 * coeff_B(coeffs, params, params['N_dc'], Np) * math.pow(v, 3) + \
                        1 / 4 * coeff_C(coeffs, params, params['N_dc']) * math.pow(v, 4))
                energy += e_
        else:
            raise TypeError("Z should have value for n_station mode")
    else:
        energy = (params['L'] / params['s'] - 1) * (1000 * params['mass'] * params['N_dc'] * math.pow(v, 2) + \
                                                    2 / params['a'] * (1 / 2 * coeff_A(coeffs, params,
                                                                                       params['N_dc'], Np) * math.pow(v,
                                                                                                                  2) + \
                                                                       1 / 3 * coeff_B(coeffs, params,
                                                                                       params['N_dc'], Np) * math.pow(v,
                                                                                                                  3) + \
                                                                       1 / 4 * coeff_C(coeffs, params,
                                                                                       params['N_dc']) * math.pow(v,
                                                                                                                  4)))
    return energy / 3600 / 1000


def E_dd(coeffs, params, v, N, Z=None, mode='n_station', Np=1):
    tmp_a = coeff_C(coeffs, params, N)
    tmp_b = coeff_B(coeffs, params, N, Np)
    tmp_c = coeff_A(coeffs, params, N, Np) - 1000 * params['mass'] * N * params['a']

    v_ = (-tmp_b + math.sqrt(tmp_b ** 2 - 4 * tmp_c * tmp_a)) / (2 * tmp_a)
    v_ = np.min([v, v_])

    if mode == 'n_station':
        if Z is not None:
            energy = 0
            for z in Z:
                e_ = 1 / 2 * 1000 * params['mass'] * z * math.pow(v_, 2) - 1 / params['a'] * (
                        1 / 2 * coeff_A(coeffs, params, params['N_dc'], Np) * math.pow(v_, 2) +
                        1 / 3 * coeff_B(coeffs, params, params['N_dc'], Np) * math.pow(v_, 3) +
                        1 / 4 * coeff_C(coeffs, params, params['N_dc']) * math.pow(v_, 4))
                energy += params['eta_reg'] * e_
        else:
            raise TypeError("Z should have value for n_station mode")
    else:
        energy = (params['L'] / params['s'] - 1) * params['eta_reg'] * (
                    1000 * params['mass'] * params['N_dc'] * math.pow(v_, 2) - \
                    2 / params['a'] * (1 / 2 * coeff_A(coeffs, params, params['N_dc'], Np) * math.pow(v_, 2) +
                                       1 / 3 * coeff_B(coeffs, params, params['N_dc'], Np) * math.pow(v_, 3) +
                                       1 / 4 * coeff_C(coeffs, params, params['N_dc']) * math.pow(v_, 4)))

    return energy / 3600 / 1000


def E_maux(coeffs, params, v, N, Z=None, mode='n_station'):
    if mode == 'n_station':
        energy = params['e_sub'] * (
                    N * (C_cc(params, v, mode=mode) + 2 * C_ca(params, v, mode=mode) + 2 * params['ts']))
        if Z is not None:
            for z in Z:
                e_ = (N - z) * (v / params['a'] + params['rs'] / v) + z * (2 * v / params['a'] + params['ts'])
                energy += params['e_sub'] * e_

    else:
        energy = params['e_sub'] * (
                    N * (C_cc(params, v, mode=mode) + 2 * C_ca(params, v, mode=mode) + 2 * params['ts']) + \
                    (N - params['N_dc']) * C_dc(params, v, mode=mode) + \
                    params['N_dc'] * (2 * (params['L'] / params['s'] - 1)) * (2 * v / params['a'] + params['ts']))
    return energy / 3600 / 1000


def E_oc(coeffs, params, v, N, mode='n_station', Np=2):
    energy = Davis_eq(coeffs, params, v, N, Np)[0] * v * C_oc(params, v, mode=mode)
    return energy / 3600 / 1000


def E_oa(coeffs, params, v, N, mode='n_station', Np=2):
    if mode == 'n_station':
        energy = 2 * (params['n_station'] - 1) * (
                    1 / 2 * 1000 * params['mass'] * N * math.pow(v, 2) + 1 / params['a'] * (
                    1 / 2 * coeff_A(coeffs, params, N, Np) * math.pow(v, 2) + \
                    1 / 3 * coeff_B(coeffs, params, N, Np) * math.pow(v, 3) + \
                    1 / 4 * coeff_C(coeffs, params, N) * math.pow(v, 4)))
    else:
        energy = 2 * params['L'] / params['s'] * (
                    1 / 2 * 1000 * params['mass'] * N * math.pow(v, 2) + 1 / params['a'] * (
                    1 / 2 * coeff_A(coeffs, params, N, Np) * math.pow(v, 2) + \
                    1 / 3 * coeff_B(coeffs, params, N, Np) * math.pow(v, 3) + \
                    1 / 4 * coeff_C(coeffs, params, N) * math.pow(v, 4)))

    return energy / 3600 / 1000


def E_od(coeffs, params, v, N, mode='n_station', Np=2):
    tmp_a = coeff_C(coeffs, params, N)
    tmp_b = coeff_B(coeffs, params, N, Np)
    tmp_c = coeff_A(coeffs, params, N, Np) - 1000 * params['mass'] * N * params['a']

    v_ = (-tmp_b + math.sqrt(tmp_b ** 2 - 4 * tmp_c * tmp_a)) / (2 * tmp_a)
    v_ = np.min([v, v_])

    if mode == 'n_station':
        energy = 2 * (params['n_station'] - 1) * params['eta_reg'] * (
                    1 / 2 * 1000 * params['mass'] * N * math.pow(v, 2) - 1 / params['a'] * (
                    1 / 2 * coeff_A(coeffs, params, N, Np) * math.pow(v_, 2) + \
                    1 / 3 * coeff_B(coeffs, params, N, Np) * math.pow(v_, 3) + \
                    1 / 4 * coeff_C(coeffs, params, N) * math.pow(v_, 4)))
    else:
        energy = 2 * params['L'] / params['s'] * params['eta_reg'] * (
                    1 / 2 * 1000 * params['mass'] * N * math.pow(v, 2) - 1 / params['a'] * (
                    1 / 2 * coeff_A(coeffs, params, N, Np) * math.pow(v_, 2) + \
                    1 / 3 * coeff_B(coeffs, params, N, Np) * math.pow(v_, 3) + \
                    1 / 4 * coeff_C(coeffs, params, N) * math.pow(v_, 4)))

    return energy / 3600 / 1000


def E_oaux(coeffs, params, v, N, mode='n_station'):
    energy = params['e_sub'] * (N * cycle_conventional(params, v, mode=mode))
    return energy / 3600 / 1000


def energy_NST(coeffs, params, v, N, Z=None, mode='n_station', regen=True, aux=True, verbose=0):
    np_main = 2
    np_decoupled = 1
    if regen:
        e_modular = E_cc(coeffs, params, v, N, Z=Z, mode=mode, Np=np_main) + E_ca(coeffs, params, v, N, Z=Z, mode=mode, Np=np_main) - \
                    E_cd(coeffs, params, v, N, Z=Z, mode=mode, Np=np_main) + E_dc(coeffs, params, v, N, Z=Z, mode=mode, Np=np_main) + \
                    E_da(coeffs, params, v, N, Z=Z, mode=mode, Np=1) - E_dd(coeffs, params, v, N, Z=Z, mode=mode, Np=1)
    else:
        e_modular = E_cc(coeffs, params, v, N, Z=Z, mode=mode, Np=np_main) + E_ca(coeffs, params, v, N, Z=Z, mode=mode, Np=np_main) + \
                    E_dc(coeffs, params, v, N, Z=Z, mode=mode, Np=np_main) + E_da(coeffs, params, v, N, Z=Z, mode=mode, Np=1)
    if aux:
        e_modular = e_modular + E_maux(coeffs, params, v, N, Z=Z, mode=mode)

    return e_modular


def energy_KTX(coeffs, params, v, N, mode='n_station', regen=True, aux=True, verbose=0):
    np_main = 2
    if regen:
        e_conventional = E_oc(coeffs, params, v, N, mode=mode, Np=np_main) + E_oa(coeffs, params, v, N, mode=mode, Np=np_main) - \
                         E_od(coeffs, params, v, N, mode=mode, Np=np_main)
    else:
        e_conventional = E_oc(coeffs, params, v, N, mode=mode, Np=np_main) + E_oa(coeffs, params, v, N, mode=mode, Np=np_main)

    if aux:
        e_conventional = e_conventional + E_oaux(coeffs, params, v, N, mode=mode)

    return e_conventional


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",  # or "DejaVu Serif" for portability
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    v = 290 / 3.6
    N = 18
    params = {'N_p': 2,  # Number of powered units
              'mass': 773.8 / 20,  # equivalent mass of each unit
              'N_b': 1.0,  # bogies per unit
              'N_pt': 1.0,  # pantographs per train
              'area': 8.304,  # frontal area (m^2)
              'Pe': math.sqrt(8.304) * 4,  # perimeter (m)
              'l_gap': 1.2,  # inter-unit distance (m)
              'l_car': (388.1 - 1.2 * (20 - 1)) / 20,  # length of each unit (m)
              'P': 13.56 * 1000,  # Power (kW)
              'C_dn': 0.15,  # air drag resistance coefficient (frontal)
              'C_db': 0.60,  # air drag resistance coefficient (bogies)
              }

    # params for energy consumption
    params['N_dc'] = 4  # average number of decoupled units at each station
    params['e_sub'] = 52.5 * 1000  # HVAC energy consumption per unit time (Watt)
    params['ts'] = 3 * 60  # dwell time at station (s)
    params['tb'] = 60  # buffer time (s)
    params['rs'] = 200
    params['L'] = 449.8 * 1000 # train length (m)
    # params['s'] = 46.4 * 1000  # stop spacing (m)
    params['s'] = params['L'] / 10  # stop spacing (m)
    params['a'] = 0.68  # acceleration / deceleration (m/s^2)
    params['eta_reg'] = 0.6  # regeneration efficiency ratio
    params['n_station'] = 11  # number of stations

    a1, a2 = 6.4, 8.0
    b1, b2, b3 = 0.18, 1.0, 0.005
    c1, c2, c3, c4, c5 = 0.6125, 0.00197, 0.0021, 0.2061, 0.2566

    coefficients = {'a1': a1,
                    'a2': a2,
                    'b1': b1,
                    'b2': b2,
                    'b3': b3,
                    'c1': c1,
                    'c2': c2,
                    'c3': c3,
                    'c4': c4,
                    'c5': c5}

    Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']

    print('==Energy conventional')

    print(E_oc(coefficients, params, v, N))
    print(E_oa(coefficients, params, v, N))
    print(E_od(coefficients, params, v, N))
    print(E_oaux(coefficients, params, v, N))

    def print_energy(coefficients, params, v, N, Z, mode):
        print('==Energy NST==')
        print('n_station')
        print('E_cc: {:.3f} kWh'.format(E_cc(coefficients, params, v, N, Z=Z, mode=mode)))
        print('E_ca: {:.3f} kWh'.format(E_ca(coefficients, params, v, N, Z=Z, mode=mode)))
        print('E_cd: {:.3f} kWh'.format(E_cd(coefficients, params, v, N, Z=Z, mode=mode)))
        print('E_dc: {:.3f} kWh'.format(E_dc(coefficients, params, v, N, Z=Z, mode=mode)))
        print('E_da: {:.3f} kWh'.format(E_da(coefficients, params, v, N, Z=Z, mode=mode)))
        print('E_dd: {:.3f} kWh'.format(E_dd(coefficients, params, v, N, Z=Z, mode=mode)))
        print('E_nt: {:.3f} kWh'.format(E_maux(coefficients, params, v, N, Z=Z, mode=mode)))

    print('==Energy NST==')
    print('n_station')
    print('E_cc: {:.3f} kWh'.format(E_cc(coefficients, params, v, N, Z=Z, mode='n_station')))
    print('E_ca: {:.3f} kWh'.format(E_ca(coefficients, params, v, N, Z=Z, mode='n_station')))
    print('E_cd: {:.3f} kWh'.format(E_cd(coefficients, params, v, N, Z=Z, mode='n_station')))
    print('E_dc: {:.3f} kWh'.format(E_dc(coefficients, params, v, N, Z=Z, mode='n_station')))
    print('E_da: {:.3f} kWh'.format(E_da(coefficients, params, v, N, Z=Z, mode='n_station')))
    print('E_dd: {:.3f} kWh'.format(E_dd(coefficients, params, v, N, Z=Z, mode='n_station')))
    print('E_nt: {:.3f} kWh'.format(E_maux(coefficients, params, v, N, Z=Z, mode='n_station')))


    print('========\nstop spacing')
    print('E_cc: {:.3f} kWh'.format(E_cc(coefficients, params, v, N, Z=Z, mode='ss')))
    print('E_ca: {:.3f} kWh'.format(E_ca(coefficients, params, v, N, Z=Z, mode='ss')))
    print('E_cd: {:.3f} kWh'.format(E_cd(coefficients, params, v, N, Z=Z, mode='ss')))
    print('E_dc: {:.3f} kWh'.format(E_dc(coefficients, params, v, N, Z=Z, mode='ss')))
    print('E_da: {:.3f} kWh'.format(E_da(coefficients, params, v, N, Z=Z, mode='ss')))
    print('E_dd: {:.3f} kWh'.format(E_dd(coefficients, params, v, N, Z=Z, mode='ss')))
    print('E_nt: {:.3f} kWh'.format(E_maux(coefficients, params, v, N, Z=Z, mode='ss')))


    print('Energy consumption for NST: {:.3f} kWh'.format(
        energy_NST(coefficients, params, v, N, Z=Z, mode='n_station', regen=True, aux=True)))
    print('Energy consumption for KTX: {:.3f} kWh'.format(
        energy_KTX(coefficients, params, v, N, mode='n_station', regen=True, aux=True)))

    """ NST vs KTX - N station mode """
    # figs, axs = plt.subplots(2, 3, figsize=(12, 6))
    # NST_energy_consumptions = [energy_NST(coefficients, params, v_ / 3.6, N,
    #                                       Z=Z, mode='n_station')
    #                            for v_ in range(100, 401, 10)]
    # KTX_energy_consumptions = [energy_NST(coefficients, params, v_ / 3.6, N,
    #                                       mode='ss')
    #                            for v_ in range(100, 401, 10)]
    #
    # figs.suptitle('NST vs. KTX Energy Consumption')
    # axs[0, 0].plot(np.arange(100, 401, 10), NST_energy_consumptions)
    # axs[0, 0].plot(np.arange(100, 401, 10), KTX_energy_consumptions)
    # axs[0, 0].legend(['ns', 'ss'])
    # axs[0, 0].set_xlabel('Crusing speed (km/h)')
    # axs[0, 0].set_ylabel('Energy (kWh)')
    # axs[0, 0].set_title('v_c')
    #
    # NST_energy_consumptions = []
    # KTX_energy_consumptions = []
    # for i in range(2, 11):
    #     params['N_dc'] = i
    #     params['N_dc'] = i
    #     Z = np.ones(params['n_station'] * 2 - 4) * i
    #     NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station',
    #                                               regen=True, aux=True))
    #     KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss',
    #                                               regen=True, aux=True))
    # params['N_dc'] = 4
    # Z = np.ones(params['n_station'] * 2) * params['N_dc']
    #
    # axs[0, 1].plot(np.arange(2, 11), NST_energy_consumptions)
    # axs[0, 1].plot(np.arange(2, 11), KTX_energy_consumptions)
    # # axs[0, 1].plot([4, 4], [31000, 33000], '--', c='grey')
    # axs[0, 1].legend(['ns', 'ss'])
    # axs[0, 1].set_xlabel('Number of decoupled units (units)')
    # axs[0, 1].set_title('N_dc')
    #
    # NST_energy_consumptions = []
    # KTX_energy_consumptions = []
    # for i in range(5, 20):
    #     params['s'] = params['L'] / (i - 1)
    #     params['n_station'] = i
    #     Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']
    #     NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station'))
    #     KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss'))
    # params['s'] = params['L'] / 11
    # params['n_station'] = 12
    # Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']
    #
    # axs[0, 2].plot(np.arange(5, 20), NST_energy_consumptions)
    # axs[0, 2].plot(np.arange(5, 20), KTX_energy_consumptions)
    # # axs[0, 2].plot([12, 12], [32000, 38000], '--', c='grey')
    # axs[0, 2].legend(['ns', 'ss'])
    # axs[0, 2].set_xlabel('Stop spacing (km)')
    # axs[0, 2].set_title('s')
    #
    # NST_energy_consumptions = []
    # KTX_energy_consumptions = []
    # for i, a_ in enumerate(np.arange(0.3, 1.55, 0.05)):
    #     params['a'] = a_
    #     NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station'))
    #     KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss'))
    # params['a'] = 0.68
    #
    # axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), NST_energy_consumptions)
    # axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), KTX_energy_consumptions)
    # # axs[1, 0].plot([0.68, 0.68], [28000, 35500], '--', c='grey')
    # axs[1, 0].legend(['ns', 'ss'])
    # axs[1, 0].set_xlabel('acceleration (m/s^2)')
    # axs[1, 0].set_title('a')
    #
    # NST_energy_consumptions = []
    # KTX_energy_consumptions = []
    # for i, e_ in enumerate(np.arange(25, 80, 2.5)):
    #     params['e_sub'] = e_ * 1000.0
    #     NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station'))
    #     KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss',
    #                                               regen=True, aux=True))
    # params['e_sub'] = 52.5 * 1000
    #
    # axs[1, 1].plot(np.arange(25, 80, 2.5), NST_energy_consumptions)
    # axs[1, 1].plot(np.arange(25, 80, 2.5), KTX_energy_consumptions)
    # # axs[1, 1].plot([52.5, 52.5], [30700, 35500], '--', c='grey')
    # axs[1, 1].legend(['ns', 'ss'])
    # axs[1, 1].set_xlabel('e_sub (kW)')
    # axs[1, 1].set_title('e_sub')
    #
    # NST_energy_consumptions = []
    # KTX_energy_consumptions = []
    # for i, e_ in enumerate(np.arange(0.1, 0.9, 0.1)):
    #     params['eta_reg'] = e_
    #     NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z,
    #                                               mode='n_station', regen=True, aux=True))
    #     KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss',
    #                                               regen=True, aux=True))
    # params['eta_reg'] = 0.6
    #
    # axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), NST_energy_consumptions)
    # axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), KTX_energy_consumptions)
    # # axs[1, 2].plot([0.2, 0.2], [30000, 40000], '--', c='grey')
    # axs[1, 2].legend(['ns', 'ss'])
    # axs[1, 2].set_xlabel('eta_reg')
    # axs[1, 2].set_title('eta_reg')
    #
    # figs.supxlabel('(Grey dashed lines indicate the default cases)', fontsize=10)
    #
    # plt.tight_layout()
    # plt.show()

    figs, axs = plt.subplots(2, 3, figsize=(12, 6))
    NST_energy_consumptions = [energy_NST(coefficients, params, v_ / 3.6, N,
                                          Z=Z, mode='n_station')
                               for v_ in range(100, 401, 10)]
    KTX_energy_consumptions = [energy_KTX(coefficients, params, v_ / 3.6, N,
                                          mode='n_station')
                               for v_ in range(100, 401, 10)]
    # params['eta_reg']=0.6

    figs.suptitle(r'NST vs. KTX Energy Consumption ($x^{1}=20, z_{i}^{k}=4 \forall i,k$}', fontsize=14)
    axs[0, 0].plot(np.arange(100, 401, 10), KTX_energy_consumptions)
    axs[0, 0].plot(np.arange(100, 401, 10), NST_energy_consumptions)
    print(KTX_energy_consumptions[-1] / KTX_energy_consumptions[20])
    print(NST_energy_consumptions[-1] / NST_energy_consumptions[20])
    # tmp_d, tmp_arg_d =
    axs[0, 0].plot([300, 300], [0, KTX_energy_consumptions[20]], linestyle='dotted', c='grey')
    # axs[0, 0].plot([300, 300], [15000, 50000], '--', c='grey')
    axs[0, 0].legend(['KTX', 'NST'])
    axs[0, 0].set_title('crusing speed')
    axs[0, 0].set_ylabel('Energy (MWh)')
    axs[0, 0].set_xlabel(r'$v_c$ (km/h)')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(2, 11):
        params['N_dc'] = i
        Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z,
                                                  mode='n_station'))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N))

    print(NST_energy_consumptions)
    params['N_dc'] = 4
    Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']

    axs[0, 1].plot(np.arange(2, 11), KTX_energy_consumptions)
    axs[0, 1].plot(np.arange(2, 11), NST_energy_consumptions)
    axs[0, 1].plot([4, 4], [0, KTX_energy_consumptions[2]], linestyle='dotted', c='grey')
    axs[0, 1].legend(['KTX', 'NST'])
    axs[0, 1].set_title('number of decoupled units')
    axs[0, 1].set_xlabel(r'$z_i^k$ (units)')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(5, 20):
        params['n_station'] = i
        Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station'))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N))
    params['n_station'] = 11
    Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']

    axs[0, 2].plot(np.arange(5, 20), KTX_energy_consumptions)
    axs[0, 2].plot(np.arange(5, 20), NST_energy_consumptions)
    axs[0, 2].plot([11, 11], [0, KTX_energy_consumptions[6]], linestyle='dotted', c='grey')
    axs[0, 2].legend(['KTX', 'NST'])
    axs[0, 2].set_xlabel('M')
    axs[0, 2].set_title('number of stations')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, a_ in enumerate(np.arange(0.3, 1.55, 0.05)):
        params['a'] = a_
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station'))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, mode='n_station'))
    params['a'] = 0.68

    axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), KTX_energy_consumptions)
    axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), NST_energy_consumptions)
    axs[1, 0].plot([0.68, 0.68], [0, KTX_energy_consumptions[8]], linestyle='dotted', c='grey')
    axs[1, 0].legend(['KTX', 'NST'])
    axs[1, 0].set_ylabel('Energy (MWh)')
    axs[1, 0].set_title('acceleration')
    axs[1, 0].set_xlabel(r'a ($m/s^2$)')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, e_ in enumerate(np.arange(25, 80, 2.5)):
        params['e_sub'] = e_ * 1000.0
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, regen=True, aux=True))
    params['e_sub'] = 52.5 * 1000

    axs[1, 1].plot(np.arange(25, 80, 2.5), KTX_energy_consumptions)
    axs[1, 1].plot(np.arange(25, 80, 2.5), NST_energy_consumptions)
    axs[1, 1].plot([52.5, 52.5], [0, KTX_energy_consumptions[11]], linestyle='dotted', c='grey')
    axs[1, 1].legend(['KTX', 'NST'])
    axs[1, 1].set_xlabel('p (kW)')
    axs[1, 1].set_title('non-traction power')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, e_ in enumerate(np.arange(0.1, 0.9, 0.1)):
        params['eta_reg'] = e_
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, regen=True, aux=True))
    params['eta_reg'] = 0.6

    axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), KTX_energy_consumptions)
    axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), NST_energy_consumptions)
    print(1 - KTX_energy_consumptions[-1] / KTX_energy_consumptions[0])
    axs[1, 2].plot([0.6, 0.6], [0, KTX_energy_consumptions[5]], linestyle='dotted', c='grey')
    axs[1, 2].legend(['KTX', 'NST'])
    axs[1, 2].set_title('regeneration efficiency')
    axs[1, 2].set_xlabel(r'$\eta_{reg}$')

    figs.supxlabel('(Grey dashed lines indicate the default cases)', fontsize=10)

    axs[0, 0].set_ylim([12000, 50000])
    axs[0, 0].set_yticks(np.arange(1.5e4, 5.2e4, 1e4))
    axs[0, 0].set_yticklabels(np.arange(15, 52, 10))
    axs[0, 1].set_ylim([12000, 50000])
    axs[0, 1].set_yticks(np.arange(1.5e4, 5.2e4, 1e4))
    axs[0, 1].set_yticklabels(np.arange(15, 52, 10))
    axs[0, 2].set_ylim([12000, 50000])
    axs[0, 2].set_yticks(np.arange(1.5e4, 5.2e4, 1e4))
    axs[0, 2].set_yticklabels(np.arange(15, 52, 10))
    axs[1, 0].set_ylim([12000, 50000])
    axs[1, 0].set_yticks(np.arange(1.5e4, 5.2e4, 1e4))
    axs[1, 0].set_yticklabels(np.arange(15, 52, 10))
    axs[1, 1].set_ylim([12000, 50000])
    axs[1, 1].set_yticks(np.arange(1.5e4, 5.2e4, 1e4))
    axs[1, 1].set_yticklabels(np.arange(15, 52, 10))
    axs[1, 2].set_ylim([12000, 50000])
    axs[1, 2].set_yticks(np.arange(1.5e4, 5.2e4, 1e4))
    axs[1, 2].set_yticklabels(np.arange(15, 52, 10))


    plt.tight_layout()
    plt.show()
    sys.exit()
    # params['eta_reg']=0.2
    # plt.savefig('Comparison_vc_Ndc_s.png')


    """ Stop Spacing """

    figs, axs = plt.subplots(2, 3, figsize=(12, 6))
    NST_energy_consumptions = [energy_NST(coefficients, params, v_ / 3.6, N,
                                          Z=Z, mode='ss')
                               for v_ in range(100, 401, 10)]
    KTX_energy_consumptions = [energy_KTX(coefficients, params, v_ / 3.6, N,
                                          mode='ss')
                               for v_ in range(100, 401, 10)]

    # params['eta_reg']=0.6

    figs.suptitle('NST vs. KTX Per-Cycle Energy Consumption')
    axs[0, 0].plot(np.arange(100, 401, 10), NST_energy_consumptions)
    axs[0, 0].plot(np.arange(100, 401, 10), KTX_energy_consumptions)
    axs[0, 0].plot([300, 300], [15000, 50000], '--', c='grey')
    axs[0, 0].legend(['NST', 'KTX'])
    axs[0, 0].set_xlabel('Crusing speed (km/h)')
    axs[0, 0].set_ylabel('Energy (kWh)')
    axs[0, 0].set_title('v_c')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(2, 11):
        params['N_dc'] = i
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss',
                                                  regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, mode='ss',
                                                  regen=True, aux=True))
    params['N_dc'] = 4
    Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']

    axs[0, 1].plot(np.arange(2, 11), NST_energy_consumptions)
    axs[0, 1].plot(np.arange(2, 11), KTX_energy_consumptions)
    axs[0, 1].plot([4, 4], [31000, 33000], '--', c='grey')
    axs[0, 1].legend(['NST', 'KTX'])
    axs[0, 1].set_xlabel('Number of decoupled units (units)')
    axs[0, 1].set_title('N_dc')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(5, 20):
        params['s'] = params['L'] / (i - 1)
        Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss'))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, mode='ss'))
    params['s'] = params['L'] / 11
    Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']


    axs[0, 2].plot(np.arange(5, 20), NST_energy_consumptions)
    axs[0, 2].plot(np.arange(5, 20), KTX_energy_consumptions)
    axs[0, 2].plot([12, 12], [32000, 38000], '--', c='grey')
    axs[0, 2].legend(['NST', 'KTX'])
    axs[0, 2].set_xlabel('Stop spacing (km)')
    axs[0, 2].set_title('s')
    axs[0, 2].set_ylim(28000, 41000)

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, a_ in enumerate(np.arange(0.3, 1.55, 0.05)):
        params['a'] = a_
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='ss'))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, mode='ss'))
    params['a'] = 0.68

    axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), NST_energy_consumptions)
    axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), KTX_energy_consumptions)
    axs[1, 0].plot([0.68, 0.68], [28000, 35500], '--', c='grey')
    axs[1, 0].legend(['NST', 'KTX'])
    axs[1, 0].set_xlabel('acceleration (m/s^2)')
    axs[1, 0].set_title('a')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, e_ in enumerate(np.arange(25, 80, 2.5)):
        params['e_sub'] = e_ * 1000.0
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss'))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, mode='ss',
                                                  regen=True, aux=True))
    params['e_sub'] = 52.5 * 1000

    axs[1, 1].plot(np.arange(25, 80, 2.5), NST_energy_consumptions)
    axs[1, 1].plot(np.arange(25, 80, 2.5), KTX_energy_consumptions)
    axs[1, 1].plot([52.5, 52.5], [30700, 35500], '--', c='grey')
    axs[1, 1].legend(['NST', 'KTX'])
    axs[1, 1].set_xlabel('e_sub (kW)')
    axs[1, 1].set_title('e_sub')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, e_ in enumerate(np.arange(0.1, 0.9, 0.1)):
        params['eta_reg'] = e_
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z,
                                                  mode='ss', regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, mode='ss',
                                                  regen=True, aux=True))
    params['eta_reg'] = 0.6

    axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), NST_energy_consumptions)
    axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), KTX_energy_consumptions)
    axs[1, 2].plot([0.2, 0.2], [30000, 40000], '--', c='grey')
    axs[1, 2].legend(['NST', 'KTX'])
    axs[1, 2].set_xlabel('eta_reg')
    axs[1, 2].set_title('eta_reg')

    figs.supxlabel('(Grey dashed lines indicate the default cases)', fontsize=10)

    plt.tight_layout()
    plt.show()

    """ Stop Spacing vs stations """

    figs, axs = plt.subplots(2, 3, figsize=(12, 6))
    NST_energy_consumptions = [energy_NST(coefficients, params, v_ / 3.6, N,
                                          Z=Z, mode='n_station')
                               for v_ in range(100, 401, 10)]
    KTX_energy_consumptions = [energy_NST(coefficients, params, v_ / 3.6, N,
                                          mode='ss')
                               for v_ in range(100, 401, 10)]

    # params['eta_reg']=0.6

    figs.suptitle('NST vs. KTX Energy Consumption')
    axs[0, 0].plot(np.arange(100, 401, 10), NST_energy_consumptions)
    axs[0, 0].plot(np.arange(100, 401, 10), KTX_energy_consumptions)
    # axs[0, 0].plot([300, 300], [15000, 50000], '--', c='grey')
    axs[0, 0].legend(['ns', 'ss'])
    axs[0, 0].set_xlabel('Crusing speed (km/h)')
    axs[0, 0].set_ylabel('Energy (kWh)')
    axs[0, 0].set_title('v_c')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(2, 11):
        params['N_dc'] = i
        Z = np.ones(params['n_station'] * 2 - 4) * i
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station',
                                                  regen=True, aux=True))
        KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss',
                                                  regen=True, aux=True))
    params['N_dc'] = 4
    Z = np.ones(params['n_station'] * 2) * params['N_dc']

    axs[0, 1].plot(np.arange(2, 11), NST_energy_consumptions)
    axs[0, 1].plot(np.arange(2, 11), KTX_energy_consumptions)
    # axs[0, 1].plot([4, 4], [31000, 33000], '--', c='grey')
    axs[0, 1].legend(['ns', 'ss'])
    axs[0, 1].set_xlabel('Number of decoupled units (units)')
    axs[0, 1].set_title('N_dc')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(5, 20):
        params['s'] = params['L'] / (i - 1)
        params['n_station'] = i
        Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station'))
        KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss'))
    params['s'] = params['L'] / 11
    params['n_station'] = 12
    Z = np.ones(params['n_station'] * 2 - 4) * params['N_dc']

    axs[0, 2].plot(np.arange(5, 20), NST_energy_consumptions)
    axs[0, 2].plot(np.arange(5, 20), KTX_energy_consumptions)
    # axs[0, 2].plot([12, 12], [32000, 38000], '--', c='grey')
    axs[0, 2].legend(['ns', 'ss'])
    axs[0, 2].set_xlabel('Stop spacing (km)')
    axs[0, 2].set_title('s')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, a_ in enumerate(np.arange(0.3, 1.55, 0.05)):
        params['a'] = a_
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station'))
        KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss'))
    params['a'] = 0.68

    axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), NST_energy_consumptions)
    axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), KTX_energy_consumptions)
    # axs[1, 0].plot([0.68, 0.68], [28000, 35500], '--', c='grey')
    axs[1, 0].legend(['ns', 'ss'])
    axs[1, 0].set_xlabel('acceleration (m/s^2)')
    axs[1, 0].set_title('a')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, e_ in enumerate(np.arange(25, 80, 2.5)):
        params['e_sub'] = e_ * 1000.0
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z, mode='n_station'))
        KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss',
                                                  regen=True, aux=True))
    params['e_sub'] = 52.5 * 1000

    axs[1, 1].plot(np.arange(25, 80, 2.5), NST_energy_consumptions)
    axs[1, 1].plot(np.arange(25, 80, 2.5), KTX_energy_consumptions)
    # axs[1, 1].plot([52.5, 52.5], [30700, 35500], '--', c='grey')
    axs[1, 1].legend(['ns', 'ss'])
    axs[1, 1].set_xlabel('e_sub (kW)')
    axs[1, 1].set_title('e_sub')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, e_ in enumerate(np.arange(0.1, 0.9, 0.1)):
        params['eta_reg'] = e_
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=Z,
                                                  mode='n_station', regen=True, aux=True))
        KTX_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss',
                                                  regen=True, aux=True))
    params['eta_reg'] = 0.6

    axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), NST_energy_consumptions)
    axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), KTX_energy_consumptions)
    # axs[1, 2].plot([0.2, 0.2], [30000, 40000], '--', c='grey')
    axs[1, 2].legend(['ns', 'ss'])
    axs[1, 2].set_xlabel('eta_reg')
    axs[1, 2].set_title('eta_reg')

    figs.supxlabel('(Grey dashed lines indicate the default cases)', fontsize=10)

    plt.tight_layout()
    plt.show()