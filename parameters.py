import math

v = 300 / 3.6
N = 20
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
params['L'] = 417.4 * 1000  # train length (m)
params['s'] = 46.4 * 1000  # stop spacing (m)
params['a'] = 0.68  # acceleration / deceleration (m/s^2)
params['eta_reg'] = 0.6  # regeneration efficiency ratio
params['n_station'] = 10  # number of stations

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