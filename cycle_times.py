import numpy as np
import math

def C_cc(params, v, mode='n_station'):
    if mode == 'n_station':
        cycle_time = 2 * params['L'] / v - 2 * (params['n_station'] - 1) * v / params['a'] - \
                     2 * (params['n_station'] - 2 ) * params['rs'] / v
    else:
        cycle_time = 2 * params['L'] / v - 2 * (params['L'] /params['s']) * v / params['a'] - \
                     2 * (params['L'] /params['s'] - 1) *  params['rs'] / v
    return cycle_time

def C_ca(params, v, mode='n_station'):
    cycle_time = 2 * v / params['a']
    return cycle_time

def C_dc(params, v, mode='n_station'):
    if mode == 'n_station':
        cycle_time = 2 * (params['n_station'] - 2) * (v / params['a'] + params['rs'] / v)
    else:
        cycle_time = 2 * (params['L'] / params['s'] - 1) * (v / (params['a']) + params['rs'] / v)
    return cycle_time

def C_da(params, v, mode='n_station'):
    if mode == 'n_station':
        cycle_time = 2 * v * (params['n_station'] - 2) / params['a']
    else:
        cycle_time = 2 * v / params['a'] * (params['L'] / params['s'] - 1)
    return cycle_time

def cycle_NST(params, v, mode='n_staion'):
    cycle_time = 2 * params['L'] / v + 2 * (params['ts'] + v / params['a'])
    return cycle_time

def C_oc(params, v, mode='n_station'):
    if mode == 'n_station':
        cycle_time = 2 * params['L'] / v - 2 * (v / params['a']) * (params['n_station'] - 1)
    else:
        cycle_time = 2 * params['L'] / v - 2 * (v / params['a']) * (params['L'] / params['s'])
    return cycle_time


def cycle_conventional(params, v, mode='n_station'):
    if mode == 'n_station':
        cycle_time = 2 * params['L'] / v + 2 * (params['n_station'] - 1) * (v / params['a'] + params['ts'])
    else:
        cycle_time = 2 * params['L'] / v + 2 * params['L'] / params['s'] * (v / params['a'] + params['ts'])
    return cycle_time


if __name__ == "__main__":
    v = 300 / 3.6
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
    params['ts'] = 5 * 60  # dwell time at station (s)
    params['tb'] = 60  # buffer time (s)
    params['rs'] = 200
    params['L'] = 417.4 * 1000  # train length (m)
    params['s'] = 46.4 * 1000  # stop spacing (m)
    params['a'] = 0.68  # acceleration / deceleration (m/s^2)
    params['eta_reg'] = 0.6  # regeneration efficiency ratio
    params['n_station'] = 10  # number of stations

    print('Cycle NST')
    print('C_cc: ', C_cc(params, v) / 3600)
    print('C_ca: ', C_ca(params, v) / 3600)
    print('C_dc: ', C_dc(params, v) / 3600)
    print('C_da: ', C_da(params, v) / 3600)
    print('cycle time: ', cycle_NST(params, v) / 3600)

    print(C_cc(params, v) / 3600 + C_dc(params, v) / 3600 + C_ca(params, v) / 3600 * 2 + 2 * params['ts'] / 3600)

    print('\nCycle conventional')
    print('C_oc: ', C_oc(params, v) / 3600)

    print('Cycle time: ', cycle_conventional(params, v) / 3600)
