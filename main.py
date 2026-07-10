import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# from scipy.special import linestyle

# from holoviews.examples.gallery.apps.bokeh.crossfilter import continuous

from energy_consumption import *
from cost import *

import sys

mpl.rcParams.update({
    "font.family": "serif",          # or "DejaVu Serif" for portability
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
params['ts'] = 3 * 60  # dwell time at station (s)
params['tb'] = 60  # buffer time (s)
params['rs'] = 200 # rail spacing (meter)
params['L'] = 438.7 * 1000  # train length (m)
params['s'] = 46.4 * 1000  # stop spacing (m)
params['a'] = 0.68  # acceleration / deceleration (m/s^2)
params['eta_reg'] = 0.6  # regeneration efficiency ratio
params['n_station'] = 11  # number of stations
params['c'] = 55 # capacity per unit (px/unit)
params['alpha'] = 1.2 # safety factor for minimum headway

# params['mu10'] = 3908703 / 30 / 365 / 17.5 * 1.5 # pro-rated life-cycle cost of each unit ($/hr)
# # params['mu11'] = params['mu10'] * 2
# params['mu11'] = 3908703 / 30 / 365 / 17.5 * 2.5
module_price = 5600000000 / 1445.44 / 30 /365 /17.5 # pro-rated life-cycle cost of each unit ($/hr)
infra_price = 65000 * 1.17 / 365 / 16 * 449.8 / 9 / 40 # infrastructure maintenance
labor_price = 70000000 / 1445.44 * 7 / 365 / 16 # employee salary
roll_unit_pr = 630000 * 1.17 / 365 / 16 / 9
roll_train_pr = 270000 * 1.17 / 365 / 16

params['mu00'] = (labor_price + roll_train_pr)
params['mu01'] = params['mu00'] * 1.5

params['mu10'] = (module_price + infra_price + roll_unit_pr)
params['mu11'] = params['mu10'] * 2

params['mu2'] = 0.14 # unit price of electricity ($/kWh)
params['mu3'] = 20.25 # value of hr ($/px-hr)
# params['mu40'] = 3000 / 17.5
# params['mu41'] = 3000 / 17.5

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

# stations = np.array([0.0, 14.9, 32.1, 103.9, 131.7, 165.1,
#                      236.5, 277.9, 291.3, 338.2, 365.7, 415.0]) * 1000
stations = np.array([0.0, 14.9, 36.9, 110.9, 139.5, 174.6,
                     248.4, 297.4, 346.4, 398.1, 438.7]) * 1000

params['n_station'] = len(stations)  # number of stations
y = 20
cont = False
print('y: ', y)
OD_out = pd.read_csv(
    'E:/Research/Non-Stop Trains/data/11stations_outbound_OD_202311.csv').values / 30 / 17.5
OD_in = pd.read_csv(
    'E:/Research/Non-Stop Trains/data/11stations_inbound_OD_202311.csv').values / 30 / 17.5
# OD_out[np.triu(np.ones_like(OD_out, dtype=bool), k=1)] = OD_out.sum() / (144 / 2 - 12)
# OD_in[np.tril(np.ones_like(OD_out, dtype=bool), k=-1)] = OD_in.sum() / (144 / 2 - 12)
# OD_out *= 3.0
# OD_in *= 3.0
print(OD_out.shape, OD_in.shape)
x_NST = min_x_NST(params, v, y, OD_out, OD_in, continuous=cont)
x_KTX = min_x_KTX(params, v, y, OD_out, OD_in, continuous=cont)
print('x (NST): ', x_NST)
print('x (KTX): ', x_KTX)
z = min_Z(params, v, y, OD_out, OD_in, continuous=cont)
print('z: ', z, sum(z))

print('KTX: Total cost: {:.2f}$/hr\n-Module cost: {:.2f}$/hr\n-Energy cost: {:.2f}$/hr\n'
      '-Waiting time cost: {:.2f}$/hr\n-In-vehicle travel time cost: {:.2f}'
      '\n-Per-train cost: {:.2f}$/hr'.format(
    *calculate_cost_KTX(coefficients, params, v, x_KTX, y,
                        stations, OD_out, OD_in, mode='n_station')))
print('NST: Total cost: {:.2f}$/hr\n-Module cost: {:.2f}$/hr\n-Energy cost: {:.2f}$/hr\n'
      '-Waiting time cost: {:.2f}$/hr\n-In-vehicle travel time cost: {:.2f}'
      '\n-Per-train cost: {:.2f}$/hr'.format(
    *calculate_cost_NST(coefficients, params, v, x_NST, y, z,
                        stations, OD_out, OD_in, mode='n_station')))

# sys.exit()


J_NST = []
module_costs_NST = []
energy_costs_NST = []
waiting_costs_NST = []
invehicle_costs_NST = []
train_costs_NST = []

J_KTX = []
module_costs_KTX = []
energy_costs_KTX = []
waiting_costs_KTX = []
invehicle_costs_KTX = []
train_costs_KTX = []

xs_NST = []
xs_KTX = []
zs = []

for i in range(1, 201):
    x_NST = min_x_NST(params, v, i, OD_out, OD_in, continuous=cont)
    x_KTX = min_x_KTX(params, v, i, OD_out, OD_in, continuous=cont)
    z = min_Z(params, v, i, OD_out, OD_in, continuous=cont)

    cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                  v=v, x=x_NST, y=i, z=z, stations=stations,
                                  OD_out=OD_out, OD_in=OD_in, mode='n_station')
    cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                  v=v, x=x_KTX, y=i, stations=stations,
                                  OD_out=OD_out, OD_in=OD_in, mode='n_station')
    xs_NST.append(x_NST)
    xs_KTX.append(x_KTX)
    zs.append(np.sum(z))
    J_NST.append(cost_NST[0])
    module_costs_NST.append(cost_NST[1])
    # print(cost[1] / (x * i + np.sum(z)))
    energy_costs_NST.append(cost_NST[2])
    waiting_costs_NST.append(cost_NST[3])
    invehicle_costs_NST.append(cost_NST[4])
    train_costs_NST.append(cost_NST[5])

    J_KTX.append(cost_KTX[0])
    module_costs_KTX.append(cost_KTX[1])
    energy_costs_KTX.append(cost_KTX[2])
    waiting_costs_KTX.append(cost_KTX[3])
    invehicle_costs_KTX.append(cost_KTX[4])
    train_costs_KTX.append(cost_KTX[5])

opt_nst = np.argmin(J_NST)
opt_ktx = np.argmin(J_KTX)
print("=====Optimal Condition=====")

print('KTX: y: {} & x: {} & total units: {}\nTotal cost: {:.2f}$/hr\n-Per-train cost: {:.2f}$/hr\n-Module cost: {:.2f}$/hr\n-Energy cost: {:.2f}$/hr\n'
      '-Waiting time cost: {:.2f}$/hr\n-In-vehicle travel time cost: {:.2f}$/hr'.format(
    opt_ktx + 1, xs_KTX[opt_ktx], (opt_ktx + 1) * xs_KTX[opt_ktx],
    J_KTX[opt_ktx],
    train_costs_KTX[opt_ktx], module_costs_KTX[opt_ktx], energy_costs_KTX[opt_ktx],
    waiting_costs_KTX[opt_ktx], invehicle_costs_KTX[opt_ktx]))


print('NST: y: {} & x: {} & total units: {}\nTotal cost: {:.2f}$/hr\n-Per-train cost: {:.2f}$/hr\n-Module cost: {:.2f}$/hr\n-Energy cost: {:.2f}$/hr\n'
      '-Waiting time cost: {:.2f}$/hr\n-In-vehicle travel time cost: {:.2f}$/hr'.format(
    opt_nst + 1, xs_NST[opt_nst], (opt_nst + 1) * xs_NST[opt_nst] + zs[opt_nst],
    J_NST[opt_nst],
    train_costs_NST[opt_nst], module_costs_NST[opt_nst], energy_costs_NST[opt_nst],
    waiting_costs_NST[opt_nst], invehicle_costs_NST[opt_nst]))

max_y_ktx = np.min(np.where(np.array([headway_KTX(params, v, i) for i in np.arange(1, 200)])
                            == params['alpha'] * v / params['a'])[0]) + 1

max_y_nst = np.min(np.where(np.array([headway_NST(params, v, i) for i in np.arange(1, 200)])
                            == params['alpha'] * v / params['a'])[0]) + 1

def print_budget_constrained():
    agency_costs_KTX = np.array(train_costs_KTX) + np.array(module_costs_KTX) + np.array(energy_costs_KTX)
    agency_costs_NST = np.array(train_costs_NST) + np.array(module_costs_NST) + np.array(energy_costs_NST)
    user_costs_KTX = np.array(waiting_costs_KTX) + np.array(invehicle_costs_KTX)
    user_costs_NST = np.array(waiting_costs_NST) + np.array(invehicle_costs_NST)

    agency_min_cost_NST, agency_min_y_NST = np.min(agency_costs_NST), np.argmin(agency_costs_NST)
    candidate_y_KTX = np.where(agency_costs_KTX < agency_min_cost_NST)[0]
    user_min_cost_KTX = np.min(user_costs_KTX[candidate_y_KTX])
    user_min_y_KTX = np.where(user_costs_KTX == user_min_cost_KTX)[0]
    # if len(user_min_y_KTX) > 1:
    #     print(user_min_y_KTX)
    #     raise ValueError('There should be 1 minimum user cost for KTX')
    # else:
    user_min_y_KTX = user_min_y_KTX[0]
    print("\n\n\n\n=====Budget Constrained Condition=====")

    print(
        'KTX: y: {} & x: {} & total units: {}\nTotal cost: {:.2f}$/hr\n-Per-train cost: {:.2f}$/hr\n-Module cost: {:.2f}$/hr\n-Energy cost: {:.2f}$/hr\n'
        '-Waiting time cost: {:.2f}$/hr\n-In-vehicle travel time cost: {:.2f}$/hr'.format(
            user_min_y_KTX + 1, xs_KTX[user_min_y_KTX],
            (user_min_y_KTX + 1) * xs_KTX[user_min_y_KTX],
            J_KTX[user_min_y_KTX],
            train_costs_KTX[user_min_y_KTX], module_costs_KTX[user_min_y_KTX],
            energy_costs_KTX[user_min_y_KTX],
            waiting_costs_KTX[user_min_y_KTX], invehicle_costs_KTX[user_min_y_KTX]))


    print(
        'NST: y: {} & x: {} & total units: {}\nTotal cost: {:.2f}$/hr\n-Per-train cost: {:.2f}$/hr\n-Module cost: {:.2f}$/hr\n-Energy cost: {:.2f}$/hr\n'
        '-Waiting time cost: {:.2f}$/hr\n-In-vehicle travel time cost: {:.2f}$/hr'.format(
            agency_min_y_NST + 1, xs_NST[agency_min_y_NST],
            (agency_min_y_NST + 1) * xs_NST[agency_min_y_NST] + zs[agency_min_y_NST],
            J_NST[agency_min_y_NST],
            train_costs_NST[agency_min_y_NST], module_costs_NST[agency_min_y_NST],
            energy_costs_NST[agency_min_y_NST],
            waiting_costs_NST[agency_min_y_NST], invehicle_costs_NST[agency_min_y_NST]))


# print_budget_constrained()

def plot_xyz():
    figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axs[0].plot(np.arange(1, 201), xs_KTX, label='KTX')
    axs[0].plot(np.arange(1, 201), xs_NST, label='NST')
    axs[0].set_title('Units per train (x)')
    axs[0].set_ylabel('Units')
    axs[0].set_xlabel('y (Number of trains)\n(a)')
    axs[0].legend()

    axs[1].plot(np.arange(1, 201), zs, c='tab:orange')
    axs[1].set_title('Sum of decoupled units (Z)')
    axs[1].set_ylabel('Units')
    axs[1].set_xlabel('y (Number of trains)\n(b)')

    axs[0].set_ylim(0, 80)
    axs[1].set_ylim(0, 80)
    # figs.suptitle('Number of units (x) and sum of decoupled units (z)\nby number of trains (y)')
    plt.tight_layout()
    plt.savefig('figures/xzs.png')
    plt.show()
    tmp_id = np.argmin(J_NST)
    print(tmp_id, xs_NST[tmp_id], zs[tmp_id])
    print(np.argmin(np.array(zs) + np.array(xs_NST) * np.arange(1, 201)))
    print(np.argmin(np.array(xs_KTX) * np.arange(1, 201)))
    print((np.array(zs) + xs_NST * np.arange(1, 201))[22])
    print((np.array(xs_KTX) * np.arange(1, 201))[0])
    # sys.exit()
    t_units_KTX = np.array(xs_KTX) * np.arange(1, 201)
    t_units_NST = np.array(zs) + np.array(xs_NST) * np.arange(1, 201)
    ktx_star, y_star_ktx = np.min(J_KTX), np.argmin(J_KTX) + 1
    nst_star, y_star_nst = np.min(J_NST), np.argmin(J_NST) + 1
    units_star_KTX = t_units_KTX[y_star_ktx-1]
    units_star_NST = t_units_NST[y_star_nst-1]


    # plt.plot(np.arange(1, 201), t_units_KTX, label='AST')
    # plt.plot(np.arange(1, 201), t_units_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx+1), t_units_KTX[:max_y_ktx], label='AST')
    plt.plot(np.arange(1, max_y_nst+1), t_units_NST[:max_y_nst], label='NST')

    plt.plot([-10, y_star_ktx], [units_star_KTX, units_star_KTX],
             linestyle='dotted', color='grey')
    plt.plot([-10, y_star_nst], [units_star_NST, units_star_NST], linestyle='dotted', color='grey')
    plt.plot([y_star_ktx, y_star_ktx], [0, units_star_KTX], linestyle='dotted', color='grey')
    plt.plot([y_star_nst, y_star_nst], [0, units_star_NST], linestyle='dotted', color='grey')

    plt.ylabel('Units')
    plt.xlabel('y (Number of trains)')

    ax = plt.gca()
    for yy in np.arange(200, 701, 100):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0, 101, 25):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    ax.set_yticks([units_star_NST, units_star_KTX, 300, 400, 500, 600, 700])
    ax.set_yticklabels([int(units_star_NST), f'{int(units_star_KTX)}\n', 300, 400, 500, 600, 700])
    ax.set_xticks([0, y_star_nst, y_star_ktx, 75, 100, 150, 200])
    ax.set_xticklabels([0, str(y_star_nst) + '\n' + r'$\mathregular{y^{1*}}$',
                        str(y_star_ktx) + '\n' + r'$\mathregular{y^{0*}}$', 75, 100, 150, 200])
    # plt.xlim(-10, 210)
    plt.xlim(-5, max_y_ktx+10)
    plt.ylim([150, 760])

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/figure4b-total units.png')
    plt.show()


def plot_total_cost():
    # plt.plot(np.arange(1, 201), J_KTX, label='AST')
    # plt.plot(np.arange(1, 201), J_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx+1), J_KTX[:max_y_ktx], label='AST')
    plt.plot(np.arange(1, max_y_nst+1), J_NST[:max_y_nst], label='NST')
    ktx_star, y_star_ktx = np.min(J_KTX), np.argmin(J_KTX) + 1
    nst_star, y_star_nst = np.min(J_NST), np.argmin(J_NST) + 1
    plt.plot([y_star_ktx, y_star_ktx], [0, ktx_star], linestyle='dotted', color='grey')
    plt.plot([y_star_nst, y_star_nst], [0, nst_star], linestyle='dotted', color='grey')

    plt.plot([-10, y_star_ktx], [ktx_star, ktx_star], linestyle='dotted', color='grey')
    plt.plot([-10, y_star_nst], [nst_star, nst_star], linestyle='dotted', color='grey')
    # plt.title('Total cost (agency + user costs)')
    plt.ylabel('Generalized cost (1,000 $/hr)')
    plt.xlabel('y (Number of trainsets)')
    plt.legend()

    ax = plt.gca()

    plt.ylim(1.3e5, 6.0e5)
    plt.xlim(-5, max_y_ktx + 10)

    ax.set_yticks([1.5e5, nst_star, ktx_star, 2.5e5, 3.5e5, 4.5e5, 5.5e5])
    ax.set_yticklabels([150, str(np.round(nst_star/1000, 1)), str(np.round(ktx_star/1000, 1)),
                        250, 350, 450, 550])
    for yy in np.arange(1.5e5, 5.51e5, 1e5):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0, 101, 25):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    ax.set_xticks([0, y_star_nst, y_star_ktx, 75, 100])

    ax.set_xticklabels([0, str(y_star_nst) + '\n' + r'$\mathregular{y^{1*}}$',
                        str(y_star_ktx) + '\n' + r'$\mathregular{y^{0*}}$', 75, 100])

    plt.tight_layout()
    plt.savefig('figures/figure4a-total costs.png')
    plt.show()

    ## xy vs. cost

    total_units_ktx = np.array(xs_KTX) * np.arange(1, 201)
    total_units_nst = (np.array(zs) + xs_NST * np.arange(1, 201))
    # plt.scatter(total_units_ktx, np.array(J_KTX), label='AST', s=5)
    # plt.scatter(total_units_nst, np.array(J_NST), label='NST', s=5)
    plt.scatter(total_units_ktx[:max_y_ktx], np.array(J_KTX)[:max_y_ktx], label='AST', s=5)
    plt.scatter(total_units_nst[:max_y_nst], np.array(J_NST)[:max_y_nst], label='NST', s=5)

    ax = plt.gca()
    ax.set_yticks(np.arange(1.5e5, 6.0e5, 1e5))
    ax.set_yticklabels(np.arange(150, 600, 100))

    t_units_ktx = y_star_ktx * xs_KTX[y_star_ktx - 1]
    t_units_nst = y_star_nst * xs_NST[y_star_nst - 1] + zs[y_star_nst - 1]
    plt.plot([t_units_ktx, t_units_ktx], [0, ktx_star],
             linestyle='dotted', color='grey')
    plt.plot([t_units_nst, t_units_nst], [0, nst_star], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_ktx], [ktx_star, ktx_star], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_nst], [nst_star, nst_star], linestyle='dotted', color='grey')

    plt.ylabel('Cost (1,000 $/hr)')
    plt.xlabel('Number of total units')

    # for yy in np.arange(1.5e5, 5.51e5, 1e5):
    #     ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    # for xx in np.arange(0, 101, 25):
    #     ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    ax.set_yticks([1.5e5, nst_star, ktx_star, 2.5e5, 3.5e5, 4.5e5, 5.5e5])
    ax.set_yticklabels([150, str(np.round(nst_star/1000, 1)),
                        str(np.round(ktx_star/1000, 1)), 250, 350, 450, 550])

    ax.set_xticks([t_units_nst, t_units_ktx,
                   250, 300, 350, 400, 500, 600, 700, 800])
    ax.set_xticklabels([int(t_units_nst), f'\n{int(t_units_ktx)}', 250, 300, 350, 400, 500, 600, 700, 800])

    # plt.xlim(170, 800)
    plt.xlim(185, 400)
    plt.ylim(1.3e5, 6.0e5)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_indiv_cost_10_100():
    figs, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axs[0, 0].plot(np.arange(11, 101), module_costs_NST[10:], label='NST')
    axs[0, 0].plot(np.arange(11, 101), module_costs_KTX[10:], label='KTX')
    axs[0, 0].set_title('Module cost')
    axs[0, 0].set_ylabel('cost ($/hr)')
    axs[0, 0].set_xlabel('y (Number of trains)')
    axs[0, 0].legend()

    axs[0, 1].plot(np.arange(11, 101), energy_costs_NST[10:], label='NST')
    axs[0, 1].plot(np.arange(11, 101), energy_costs_KTX[10:], label='KTX')
    axs[0, 1].set_title('Energy cost')
    axs[0, 1].set_ylabel('cost ($/hr)')
    axs[0, 1].set_xlabel('y (Number of trains)')
    axs[0, 1].legend()

    axs[1, 0].plot(np.arange(11, 101), waiting_costs_NST[10:], label='NST')
    axs[1, 0].plot(np.arange(11, 101), waiting_costs_KTX[10:], label='KTX')
    axs[1, 0].set_title('Waiting time cost')
    axs[1, 0].set_ylabel('cost ($/hr)')
    axs[1, 0].set_xlabel('y (Number of trains)')
    axs[1, 0].legend()

    axs[1, 1].plot(np.arange(11, 101), invehicle_costs_NST[10:], label='NST')
    axs[1, 1].plot(np.arange(11, 101), invehicle_costs_KTX[10:], label='KTX')
    axs[1, 1].set_title('In-vehicle travel time cost')
    axs[1, 1].set_ylabel('cost ($/hr)')
    axs[1, 1].set_xlabel('y (Number of trains)')
    axs[1, 1].legend()

    # axs[0, 0].set_ylim([29000, 49000])
    # axs[0, 1].set_ylim([0, 20000])
    # axs[1, 0].set_ylim([11000, 31000])
    # axs[1, 1].set_ylim([11000, 31000])

    figs.suptitle('Costs (Total, agency, energy, and user)\nby number of trains (y)')
    plt.tight_layout()
    plt.savefig('figures/costs_10_100.png')
    plt.show()


def plot_agency_vs_user():
    figs, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    agency_costs_KTX = np.array(train_costs_KTX) + np.array(module_costs_KTX) + np.array(energy_costs_KTX)
    agency_costs_NST = np.array(train_costs_NST) + np.array(module_costs_NST) + np.array(energy_costs_NST)
    axs[0].plot(np.arange(1, 201), agency_costs_KTX, label='KTX')
    axs[0].plot(np.arange(1, 201), agency_costs_NST, label='NST')

    axs[0].plot([opt_ktx + 1, opt_ktx + 1], [0, agency_costs_KTX[opt_ktx]], '--', c='grey')
    axs[0].plot([0, opt_ktx + 1], [agency_costs_KTX[opt_ktx], agency_costs_KTX[opt_ktx]], '--', c='grey')
    axs[0].plot([opt_nst + 1, opt_nst + 1], [0, agency_costs_NST[opt_nst]], '--', c='grey')
    axs[0].plot([0, opt_nst + 1], [agency_costs_NST[opt_nst], agency_costs_NST[opt_nst]], '--', c='grey')
    axs[0].set_title('agency cost')
    axs[0].set_ylabel('cost (1,000 $/hr)')
    axs[0].set_xlabel('y (Number of trains)')
    axs[0].legend()

    user_costs_KTX = np.array(waiting_costs_KTX) + np.array(invehicle_costs_KTX)
    user_costs_NST = np.array(waiting_costs_NST) + np.array(invehicle_costs_NST)

    axs[1].plot(np.arange(1, 201), user_costs_KTX, label='KTX')
    axs[1].plot(np.arange(1, 201), user_costs_NST, label='NST')
    axs[1].plot([opt_ktx + 1, opt_ktx + 1], [0, user_costs_KTX[opt_ktx]], '--', c='grey')
    axs[1].plot([0, opt_ktx + 1],
                   [user_costs_KTX[opt_ktx], user_costs_KTX[opt_ktx]], '--', c='grey')
    axs[1].plot([opt_nst + 1, opt_nst + 1], [0, user_costs_NST[opt_nst]], '--', c='grey')
    axs[1].plot([0, opt_nst + 1],
                   [user_costs_NST[opt_nst], user_costs_NST[opt_nst]], '--', c='grey')

    axs[1].set_title('user cost')
    axs[1].set_ylabel('cost (1,000 $/hr)')
    axs[1].set_xlabel('y (Number of trains)')
    axs[1].legend()

    ### Axis for the main figure
    ktx_star, y_star_ktx = np.min(J_KTX), np.argmin(J_KTX) + 1
    nst_star, y_star_nst = np.min(J_NST), np.argmin(J_NST) + 1

    axs[0].set_xticks([0, y_star_nst, y_star_ktx, 75, 100, 150, 200])
    axs[0].set_xticklabels([0, str(y_star_nst) + '\n' + r'$\mathregular{y^{1*}}$',
                              str(y_star_ktx) + '\n' + r'$\mathregular{y^{0*}}$', 75, 100, 150, 200])
    axs[1].set_xticks([0, y_star_nst, y_star_ktx, 75, 100, 150, 200])
    axs[1].set_xticklabels([0, str(y_star_nst) + '\n' + r'$\mathregular{y^{1*}}$',
                               str(y_star_ktx) + '\n' + r'$\mathregular{y^{0*}}$', 75, 100, 150, 200])


    y_max = 350000
    axs[0].set_ylim([15000, y_max])
    axs[0].set_yticks(np.arange(5e4, y_max + 1, 5e4))
    axs[0].set_yticklabels(np.arange(50, y_max/1000 + 1, 50))
    axs[1].set_ylim([0, y_max])
    axs[1].set_yticks(np.arange(5e4, y_max + 1, 5e4))
    axs[1].set_yticklabels(np.arange(50, y_max/1000 + 1, 50))



    # figs.suptitle('Costs (Total, , energy, and user)\nby number of trains (y)', fontsize=15)
    plt.tight_layout()
    plt.savefig('figures/agency_vs_user_costs.png')
    plt.show()

    # plt.plot(np.arange(1, 201), agency_costs_KTX, label='KTX')
    # plt.plot(np.arange(1, 201), agency_costs_NST, label='NST')
    # # plt.plot([ + 1, opt_ktx + 1], [0, agency_costs_KTX[opt_ktx]], '--', c='grey')
    # plt.plot([0, 200], [np.min(agency_costs_NST), np.min(agency_costs_NST)], '--', c='grey')
    # # plt.plot([opt_nst + 1, opt_nst + 1], [0, agency_costs_NST[opt_nst]], '--', c='grey')
    # # plt.plot([0, opt_nst + 1], [agency_costs_NST[opt_nst], agency_costs_NST[opt_nst]], '--', c='grey')
    # plt.title('agency cost')
    # plt.ylabel('cost (1,000 $/hr)')
    # plt.xlabel('y (Number of trains)')
    # plt.legend()

    # figs, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ktx_star, y_star_ktx = np.min(J_KTX), np.argmin(J_KTX)
    min_agency_nst, min_agency_nst_y = np.min(agency_costs_NST), np.argmin(agency_costs_NST)

    # plt.plot(np.arange(1, 201), agency_costs_KTX, label='AST')
    # plt.plot(np.arange(1, 201), agency_costs_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx+1), agency_costs_KTX[:max_y_ktx], label='AST')
    plt.plot(np.arange(1, max_y_nst+1), agency_costs_NST[:max_y_nst], label='NST')

    plt.plot([-10, 210],
                [min_agency_nst, min_agency_nst], '-', c='black')
    plt.plot([min_agency_nst_y + 1, min_agency_nst_y + 1],
             [0, min_agency_nst], '--', c='grey')

    idx_ktx = np.where(agency_costs_KTX <= min_agency_nst)[0]
    idx_nst = np.where(agency_costs_NST <= min_agency_nst)[0]
    plt.plot(idx_ktx + 1, agency_costs_KTX[idx_ktx], '.', c='blue')
    plt.plot(idx_nst + 1, agency_costs_NST[idx_nst], 'o', c='red')
    # plt.plot([np.max(idx_ktx) + 1, np.max(idx_ktx) + 1],
    #          [0, min_agency_nst], '--', c='grey')
    # plt.plot([idx_nst[0] + 1, idx_nst[0] + 1], [0, min_agency_nst], '--', c='grey')
    plt.title('Operator cost')
    plt.ylabel('operator cost (1,000 $/hr)')
    plt.xlabel('y (Number of trains)')
    plt.legend()

    y_max = 350000
    ax = plt.gca()
    # ax.set_yticks(np.arange(5e4, y_max + 1, 5e4))
    # ax.set_yticklabels(np.arange(50, int(y_max / 1000) + 1, 50))
    ax.set_yticks([min_agency_nst,
                   5e4, 10e4, 15e4, 20e4, 25e4, 30e4, 35e4])

    ax.set_yticklabels([f'{np.round(min_agency_nst / 1000, 1)}',
                        50, 100, 150, 200, 250, 300, 350])
    ax.set_xticks([0, idx_nst[0] + 1, 50, 100, 150, 200])
    plt.ylim([15000, y_max])
    plt.xlim([-5, max_y_ktx+10])
    plt.tight_layout()
    plt.show()


    # plt.plot(np.arange(1, 201), user_costs_KTX, label='AST')
    # plt.plot(np.arange(1, 201), user_costs_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx + 1), user_costs_KTX[:max_y_ktx], label='AST')
    plt.plot(np.arange(1, max_y_nst + 1), user_costs_NST[:max_y_nst], label='NST')

    plt.plot(idx_ktx + 1, user_costs_KTX[idx_ktx], '.', c='blue')
    plt.plot(idx_nst + 1, user_costs_NST[idx_nst], 'o', c='red')

    plt.plot([-10, np.max(idx_ktx)+1],
             [user_costs_KTX[np.max(idx_ktx)], user_costs_KTX[np.max(idx_ktx)]], '--', c='grey')
    plt.plot([-10, idx_nst[0] + 1],
             [user_costs_NST[idx_nst[0]], user_costs_NST[idx_nst[0]]], '--', c='grey')
    plt.plot([np.max(idx_ktx)+1, np.max(idx_ktx)+1],
             [0, user_costs_KTX[np.max(idx_ktx)]], '--', c='grey')
    plt.plot([idx_nst[0] + 1, idx_nst[0] + 1],
             [0, user_costs_NST[idx_nst[0]]], '--', c='grey')

    plt.title('User cost')
    plt.ylabel('user cost (1,000 $/hr)')
    plt.xlabel('y (Number of trains)')
    plt.legend()

    y_max = 350000
    ax = plt.gca()
    # ax.set_yticks(np.arange(5e4, y_max + 1, 5e4))
    ax.set_yticks([5e4, 10e4, 15e4,
                   user_costs_NST[idx_nst[0]], user_costs_KTX[np.max(idx_ktx)],
                   20e4, 25e4, 30e4, 35e4])
    # ax.set_yticklabels(np.arange(50, int(y_max / 1000) + 1, 50))
    ax.set_yticklabels([50, 100, 150,
                        f'{np.round(user_costs_NST[idx_nst[0]] / 1000, 1)}',
                        f'{np.round(user_costs_KTX[np.max(idx_ktx)] / 1000, 1)}',
                        200, 250, 300, 350])
    ax.set_xticks([0, idx_nst[0]+1, 50, np.max(idx_ktx)+1, 100, 150, 200])
    plt.ylim([15000, y_max])
    # plt.xlim([-10, 210])
    plt.xlim([-5, max_y_ktx+10])

    plt.tight_layout()
    plt.show()


    # plt.plot(np.arange(1, 201), J_KTX, label='AST')
    # plt.plot(np.arange(1, 201), J_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx + 1), J_KTX[:max_y_ktx], label='AST')
    plt.plot(np.arange(1, max_y_nst + 1), J_NST[:max_y_nst], label='NST')

    plt.plot(idx_ktx + 1, np.array(J_KTX)[idx_ktx], '.', c='blue')
    plt.plot(idx_nst + 1, np.array(J_NST)[idx_nst], 'o', c='red')

    candidate_J_KTX = np.array(J_KTX)[idx_ktx]
    budget_min, budget_y = np.min(candidate_J_KTX), idx_ktx[np.argmin(candidate_J_KTX)]

    plt.plot([-10, budget_y+1],
             [budget_min, budget_min], '--', c='grey')
    plt.plot([-10, idx_nst[0] + 1],
             [J_NST[idx_nst[0]], J_NST[idx_nst[0]]], '--', c='grey')
    plt.plot([budget_y+1, budget_y+1],
             [0, budget_min], '--', c='grey')
    plt.plot([idx_nst[0] + 1, idx_nst[0] + 1],
             [0, J_NST[idx_nst[0]]], '--', c='grey')

    plt.title('Generalized cost')
    plt.ylabel('generalized cost (1,000 $/hr)')
    plt.xlabel('y (Number of trains)')
    plt.legend()

    y_max = 350000
    ax = plt.gca()
    # ax.set_yticks(np.arange(5e4, y_max + 1, 5e4))
    ax.set_yticks([5e4, 10e4, 15e4, 20e4,
                   J_NST[idx_nst[0]], budget_min,
                   25e4, 30e4, 35e4])
    # ax.set_yticklabels(np.arange(50, int(y_max / 1000) + 1, 50))
    ax.set_yticklabels([50, 100, 150, 200,
                        f'{np.round(J_NST[idx_nst[0]] / 1000, 1)}\n',
                        f'{np.round(budget_min / 1000, 1)}\n',
                        250, 300, 350])
    ax.set_xticks([0, idx_nst[0] + 1, budget_y + 1, 100, 150, 200])
    plt.ylim([15000, y_max])
    # plt.xlim([-10, 210])
    plt.xlim([-5, max_y_ktx+10])
    plt.tight_layout()
    plt.show()


def plot_operator_vs_user_v2():
    agency_costs_KTX = np.array(train_costs_KTX) + np.array(module_costs_KTX) + np.array(energy_costs_KTX)
    agency_costs_NST = np.array(train_costs_NST) + np.array(module_costs_NST) + np.array(energy_costs_NST)
    user_costs_KTX = np.array(waiting_costs_KTX) + np.array(invehicle_costs_KTX)
    user_costs_NST = np.array(waiting_costs_NST) + np.array(invehicle_costs_NST)

    plt.scatter(agency_costs_KTX[:max_y_ktx], user_costs_KTX[:max_y_ktx], label='AST')
    plt.scatter(agency_costs_NST[:max_y_nst], user_costs_NST[:max_y_nst], label='NST')

    plt.plot(agency_costs_KTX[opt_ktx], user_costs_KTX[opt_ktx],
             '*', c='blue', markersize=10)
    plt.plot(agency_costs_NST[opt_nst], user_costs_NST[opt_nst],
             '*', c='red', markersize=10)

    plt.xlabel('Operator cost ($1,000/hr)')
    plt.ylabel('User cost ($1,000/hr)')

    ax = plt.gca()
    ax.set_yticks(np.arange(1e5, 9e5+1, 5e4))
    ax.set_yticklabels(np.arange(100, 910, 50))
    ax.set_xticks(np.arange(2e4, 12e4+1, 1e4))
    ax.set_xticklabels(np.arange(20, 121, 10))
    plt.ylim([0.9e5, 3.5e5])
    x0, x1 = 1.9e4, 7.8e4
    ax.set_xlim([x0, x1])
    xA_end = np.min(agency_costs_NST[:max_y_nst])
    xB_end = np.max(agency_costs_KTX[:max_y_ktx])
    ax.axvspan(x0, xA_end, facecolor='C0', alpha=0.10, zorder=0)  # A
    ax.axvspan(xA_end, xB_end, facecolor='C1', alpha=0.10, zorder=0)  # B
    ax.axvspan(xB_end, x1, facecolor='C2', alpha=0.10, zorder=0)  # C

    ax.axvline(xA_end, color='k', linewidth=1.0, alpha=0.5, zorder=1)
    ax.axvline(xB_end, color='k', linewidth=1.0, alpha=0.5, zorder=1)

    y_top = ax.get_ylim()[1]
    ax.text((x0 + xA_end) / 2, 0.98 * y_top, "A", ha="center", va="top", fontsize=12)
    ax.text((xA_end + xB_end) / 2, 0.98 * y_top, "B", ha="center", va="top", fontsize=12)
    ax.text((xB_end + x1) / 2.07, 0.98 * y_top, "C", ha="center", va="top", fontsize=12)

    plt.legend()
    for yy in np.arange(1.5e5, 9.51e5, 5e4):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(2e4, 12e4+1, 1e4):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure8-agency_vs_user_cost.png')
    plt.show()


def plot_indiv_cost():
    ktx_star, y_star_ktx = np.min(J_KTX), np.argmin(J_KTX) + 1
    nst_star, y_star_nst = np.min(J_NST), np.argmin(J_NST) + 1
    y_max = 50000

    # plt.plot(np.arange(1, 201), train_costs_KTX, label='AST')
    # plt.plot(np.arange(1, 201), train_costs_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx+1), train_costs_KTX[:max_y_ktx], label='AST')
    plt.plot(np.arange(1, max_y_nst+1), train_costs_NST[:max_y_nst], label='NST')

    plt.plot([opt_ktx + 1, opt_ktx + 1], [0, train_costs_KTX[opt_ktx]], '--', c='grey')
    plt.plot([-10, opt_ktx + 1], [train_costs_KTX[opt_ktx], train_costs_KTX[opt_ktx]], '--', c='grey')
    plt.plot([opt_nst + 1, opt_nst + 1], [0, train_costs_NST[opt_nst]], '--', c='grey')
    plt.plot([-10, opt_nst + 1], [train_costs_NST[opt_nst], train_costs_NST[opt_nst]], '--', c='grey')
    plt.title('train cost')
    plt.ylabel('cost (1,000 $/hr)')
    plt.xlabel('y (Number of trains)')
    plt.legend()

    ax = plt.gca()
    ax.set_xticks([0, y_star_nst, y_star_ktx, 75, 100, 150, 200])
    ax.set_xticklabels([0, str(y_star_nst) + '\n' + r'$\mathregular{y^{1*}}$',
                               str(y_star_ktx) + '\n' + r'$\mathregular{y^{0*}}$', 75, 100, 150, 200])

    train_star_ktx = train_costs_KTX[y_star_ktx-1]
    train_star_nst = train_costs_NST[y_star_nst-1]
    ax.set_yticks([0, train_star_nst, train_star_ktx, 10e3, 20e3, 30e3, 40e3, 50e3])
    ax.set_yticklabels([0,
                        f'\n{np.round(train_star_nst / 1000, 1)}',
                        f'{np.round(train_star_ktx / 1000, 1)}',
                        10, 20, 30, 40, 50])
    ax.set_ylim([0, y_max])
    # ax.set_xlim([-10, 210])
    ax.set_xlim([-5, max_y_ktx + 10])
    for yy in np.arange(1e4, 5.1e4, 1e4):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0, 101, 25):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    plt.tight_layout()
    plt.savefig('figures/figure7a-train_cost.png')
    plt.show()

    # plt.plot(np.arange(1, 201), module_costs_KTX, label='KTX')
    # plt.plot(np.arange(1, 201), module_costs_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx+1), module_costs_KTX[:max_y_ktx], label='KTX')
    plt.plot(np.arange(1, max_y_nst+1), module_costs_NST[:max_y_nst], label='NST')
    plt.plot([opt_ktx + 1, opt_ktx + 1], [0, module_costs_KTX[opt_ktx]], '--', c='grey')
    plt.plot([-10, opt_ktx + 1],
                   [module_costs_KTX[opt_ktx], module_costs_KTX[opt_ktx]], '--', c='grey')
    plt.plot([opt_nst + 1, opt_nst + 1], [0, module_costs_NST[opt_nst]], '--', c='grey')
    plt.plot([-10, opt_nst + 1],
                   [module_costs_NST[opt_nst], module_costs_NST[opt_nst]], '--', c='grey')

    plt.title('unit cost')
    plt.ylabel('cost (1,000 $/hr)')
    plt.xlabel('y (Number of trains)')
    plt.legend()

    ax = plt.gca()
    ax.set_xticks([0, y_star_nst, y_star_ktx, 75, 100, 150, 200])
    ax.set_xticklabels([0, str(y_star_nst) + '\n' + r'$\mathregular{y^{1*}}$',
                               str(y_star_ktx) + '\n' + r'$\mathregular{y^{0*}}$', 75, 100, 150, 200])

    module_star_ktx = module_costs_KTX[y_star_ktx-1]
    module_star_nst = module_costs_NST[y_star_nst-1]
    ax.set_yticks([0, module_star_ktx, module_star_nst, 30e3, 40e3, 50e3])
    ax.set_yticklabels([0,
                        f'{np.round(module_star_ktx / 1000, 1)}',
                        f'{np.round(module_star_nst / 1000, 1)}',
                        30, 40, 50])
    ax.set_ylim([0, y_max])
    # ax.set_xlim([-10, 210])
    ax.set_xlim([-5, max_y_ktx+10])
    for yy in np.arange(1e4, 5.1e4, 1e4):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0, 101, 25):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure7b-module_cost.png')
    plt.show()

    # plt.plot(np.arange(1, 201), energy_costs_KTX, label='KTX')
    # plt.plot(np.arange(1, 201), energy_costs_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx+1), energy_costs_KTX[:max_y_ktx], label='KTX')
    plt.plot(np.arange(1, max_y_nst+1), energy_costs_NST[:max_y_nst], label='NST')
    plt.plot([opt_ktx + 1, opt_ktx + 1], [0, energy_costs_KTX[opt_ktx]], '--', c='grey')
    plt.plot([-10, opt_ktx + 1], [energy_costs_KTX[opt_ktx], energy_costs_KTX[opt_ktx]], '--', c='grey')
    plt.plot([opt_nst + 1, opt_nst + 1], [0, energy_costs_NST[opt_nst]], '--', c='grey')
    plt.plot([-10, opt_nst + 1], [energy_costs_NST[opt_nst], energy_costs_NST[opt_nst]], '--', c='grey')

    plt.title('energy cost')
    plt.ylabel('cost (1,000 $/hr)')
    plt.xlabel('y (Number of trains)')
    plt.legend()

    ax = plt.gca()
    ax.set_xticks([0, y_star_nst, y_star_ktx, 75, 100, 150, 200])
    ax.set_xticklabels([0, str(y_star_nst) + '\n' + r'$\mathregular{y^{1*}}$',
                               str(y_star_ktx) + '\n' + r'$\mathregular{y^{0*}}$', 75, 100, 150, 200])

    energy_star_ktx = energy_costs_KTX[y_star_ktx-1]
    energy_star_nst = energy_costs_NST[y_star_nst-1]
    ax.set_yticks([0, 10e3, energy_star_ktx, energy_star_nst, 20e3, 30e3, 40e3, 50e3])
    ax.set_yticklabels([0, 10,
                        f'\n{np.round(energy_star_ktx / 1000, 1)}',
                        f'{np.round(energy_star_nst / 1000, 1)}\n',
                        20, 30, 40, 50])
    ax.set_ylim([0, y_max])
    # ax.set_xlim([-10, 210])
    ax.set_xlim([-5, max_y_ktx+10])

    for yy in np.arange(1e4, 5.1e4, 1e4):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0, 101, 25):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure7c-energy_cost.png')
    plt.show()

    # plt.plot(np.arange(1, 201), waiting_costs_KTX, label='KTX')
    # plt.plot(np.arange(1, 201), waiting_costs_NST, label='NST')
    plt.plot(np.arange(1, max_y_ktx+1), waiting_costs_KTX[:max_y_ktx], label='KTX')
    plt.plot(np.arange(1, max_y_nst+1), waiting_costs_NST[:max_y_nst], label='NST')

    plt.plot([opt_ktx + 1, opt_ktx + 1], [0, waiting_costs_KTX[opt_ktx]], '--', c='grey')
    plt.plot([0, opt_ktx + 1], [waiting_costs_KTX[opt_ktx], waiting_costs_KTX[opt_ktx]], '--', c='grey')
    plt.plot([opt_nst + 1, opt_nst + 1], [0, waiting_costs_NST[opt_nst]], '--', c='grey')
    plt.plot([0, opt_nst + 1], [waiting_costs_NST[opt_nst], waiting_costs_NST[opt_nst]], '--', c='grey')

    plt.title('waiting time cost')
    plt.ylabel('cost (1,000 $/hr)')
    plt.xlabel('y (Number of trains)')
    plt.legend()

    ax = plt.gca()
    ax.set_xticks([0, y_star_nst, y_star_ktx, 75, 100, 150, 200])
    ax.set_xticklabels([0, str(y_star_nst) + '\n' + r'$\mathregular{y^{1*}}$',
                               str(y_star_ktx) + '\n' + r'$\mathregular{y^{0*}}$', 75, 100, 150, 200])

    waiting_star_ktx = waiting_costs_KTX[y_star_ktx-1]
    waiting_star_nst = waiting_costs_NST[y_star_nst-1]
    ax.set_yticks([0, 10e3, waiting_star_ktx, waiting_star_nst, 20e3, 30e3, 40e3, 50e3])
    ax.set_yticklabels([0, 10,
                        f'{np.round(waiting_star_ktx / 1000, 1)}',
                        f'{np.round(waiting_star_nst / 1000, 1)}',
                        20, 30, 40, 50])
    ax.set_ylim([0, y_max])
    # ax.set_xlim([-10, 210])
    ax.set_xlim([-5, max_y_ktx+10])

    for yy in np.arange(1e4, 5.1e4, 1e4):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0, 101, 25):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure7d-waiting_time_cost.png')
    plt.show()

    #### t_units vs cost
    total_units_ktx = np.array(xs_KTX) * np.arange(1, 201)
    total_units_nst = (np.array(zs) + xs_NST * np.arange(1, 201))
    plt.scatter(total_units_ktx, train_costs_KTX, label='AST', s=5)
    plt.scatter(total_units_nst, train_costs_NST, label='NST', s=5)
    plt.xlim(170, 800)
    plt.ylim(0, y_max)
    ax = plt.gca()

    t_units_ktx = y_star_ktx * xs_KTX[y_star_ktx - 1]
    t_units_nst = y_star_nst * xs_NST[y_star_nst - 1] + zs[y_star_nst - 1]

    plt.plot([t_units_ktx, t_units_ktx], [0, train_star_ktx],
             linestyle='dotted', color='grey')
    plt.plot([t_units_nst, t_units_nst], [0, train_star_nst], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_ktx], [train_star_ktx, train_star_ktx], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_nst], [train_star_nst, train_star_nst], linestyle='dotted', color='grey')

    plt.ylabel('Cost (1,000 $/hr)')
    plt.xlabel('Number of total units')
    plt.title('train cost')

    ax.set_yticks([0, train_star_nst, train_star_ktx, 10e3, 20e3, 30e3, 40e3, 50e3])
    ax.set_yticklabels([0,
                        f'\n{np.round(train_star_nst/1000, 1)}',
                        f'{np.round(train_star_ktx/1000, 1)}',
                       10, 20, 30, 40, 50])

    ax.set_xticks([t_units_nst, t_units_ktx,
                   300, 400, 500, 600, 700, 800])
    ax.set_xticklabels([int(t_units_nst), f'\n{int(t_units_ktx)}', 300, 400, 500, 600, 700, 800])

    plt.legend()
    plt.tight_layout()
    plt.show()


    plt.scatter(total_units_ktx, module_costs_KTX, label='AST', s=5)
    plt.scatter(total_units_nst, module_costs_NST, label='NST', s=5)
    plt.xlim(170, 800)
    plt.ylim(0, y_max)
    ax = plt.gca()

    plt.plot([t_units_ktx, t_units_ktx], [0, module_star_ktx],
             linestyle='dotted', color='grey')
    plt.plot([t_units_nst, t_units_nst], [0, module_star_nst], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_ktx], [module_star_ktx, module_star_ktx], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_nst], [module_star_nst, module_star_nst], linestyle='dotted', color='grey')

    plt.ylabel('Cost (1,000 $/hr)')
    plt.xlabel('Number of total units')
    plt.title('unit cost')

    ax.set_yticks([0, module_star_ktx, module_star_nst, 30e3, 40e3, 50e3])
    ax.set_yticklabels([0,
                        f'{np.round(energy_star_ktx / 1000, 1)}',
                        f'{np.round(energy_star_nst / 1000, 1)}',
                        30, 40, 50])

    ax.set_xticks([t_units_nst, t_units_ktx,
                   300, 400, 500, 600, 700, 800])
    ax.set_xticklabels([int(t_units_nst), f'\n{int(t_units_ktx)}', 300, 400, 500, 600, 700, 800])

    plt.legend()
    plt.tight_layout()
    plt.show()


    plt.scatter(total_units_ktx, energy_costs_KTX, label='AST', s=5)
    plt.scatter(total_units_nst, energy_costs_NST, label='NST', s=5)
    plt.xlim(170, 800)
    plt.ylim(0, y_max)
    ax = plt.gca()

    plt.plot([t_units_ktx, t_units_ktx], [0, energy_star_ktx],
             linestyle='dotted', color='grey')
    plt.plot([t_units_nst, t_units_nst], [0, energy_star_nst], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_ktx], [energy_star_ktx, energy_star_ktx], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_nst], [energy_star_nst, energy_star_nst], linestyle='dotted', color='grey')

    plt.ylabel('Cost (1,000 $/hr)')
    plt.xlabel('Number of total units')
    plt.title('energy cost')

    ax.set_yticks([0, 10e3, energy_star_ktx, energy_star_nst, 20e3, 30e3, 40e3, 50e3])
    ax.set_yticklabels([0, 10,
                        f'\n{np.round(energy_star_ktx/1000, 1)}',
                        f'{np.round(energy_star_nst/1000, 1)}',
                       20, 30, 40, 50])

    ax.set_xticks([t_units_nst, t_units_ktx,
                   300, 400, 500, 600, 700, 800])
    ax.set_xticklabels([int(t_units_nst), f'\n{int(t_units_ktx)}', 300, 400, 500, 600, 700, 800])

    plt.legend()
    plt.tight_layout()
    plt.show()


    plt.scatter(total_units_ktx, waiting_costs_KTX, label='AST', s=5)
    plt.scatter(total_units_nst, waiting_costs_NST, label='NST', s=5)
    plt.xlim(170, 800)
    plt.ylim(0, y_max)
    ax = plt.gca()

    plt.plot([t_units_ktx, t_units_ktx], [0, waiting_star_ktx],
             linestyle='dotted', color='grey')
    plt.plot([t_units_nst, t_units_nst], [0, waiting_star_nst], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_ktx], [waiting_star_ktx, waiting_star_ktx], linestyle='dotted', color='grey')
    plt.plot([-10, t_units_nst], [waiting_star_nst, waiting_star_nst], linestyle='dotted', color='grey')

    plt.ylabel('Cost (1,000 $/hr)')
    plt.xlabel('Number of total units')
    plt.title('waiting time cost')

    ax.set_yticks([0, 10e3, waiting_star_ktx, waiting_star_nst, 20e3, 30e3, 40e3, 50e3])
    ax.set_yticklabels([0, 10,
                        f'{np.round(waiting_star_ktx/1000, 1)}',
                        f'{np.round(waiting_star_nst/1000, 1)}',
                       20, 30, 40, 50])

    ax.set_xticks([t_units_nst, t_units_ktx,
                   300, 400, 500, 600, 700, 800])
    ax.set_xticklabels([int(t_units_nst), f'{int(t_units_ktx)}', 300, 400, 500, 600, 700, 800])

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_indiv_barplot():
    # ratio
    ektx = [train_costs_KTX[opt_ktx], module_costs_KTX[opt_ktx], energy_costs_KTX[opt_ktx],
            waiting_costs_KTX[opt_ktx], invehicle_costs_KTX[opt_ktx]]

    enst = [train_costs_NST[opt_nst], module_costs_NST[opt_nst], energy_costs_NST[opt_nst],
            waiting_costs_NST[opt_nst], invehicle_costs_NST[opt_nst]]
    ektx = np.array(ektx)
    enst = np.array(enst)

    fig, ax = plt.subplots()
    ax.bar(np.arange(5) - 0.2, ektx / J_KTX[opt_ktx], width=0.4, label='AST')
    ax.bar(np.arange(5) + 0.2, enst / J_NST[opt_nst], width=0.4, label='NST')

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['Train', 'Module', 'Energy', 'Waiting', 'InVehicle'])#, rotation=45)
    ax.set_ylabel('Cost ratio')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()

    # Absolute
    ektx = [train_costs_KTX[opt_ktx], module_costs_KTX[opt_ktx], energy_costs_KTX[opt_ktx],
            waiting_costs_KTX[opt_ktx], invehicle_costs_KTX[opt_ktx]]

    enst = [train_costs_NST[opt_nst], module_costs_NST[opt_nst], energy_costs_NST[opt_nst],
            waiting_costs_NST[opt_nst], invehicle_costs_NST[opt_nst]]
    ektx = np.array(ektx)
    enst = np.array(enst)

    fig, ax = plt.subplots()
    ax.bar(np.arange(5) - 0.2, ektx, width=0.4, label='AST')
    ax.bar(np.arange(5) + 0.2, enst, width=0.4, label='NST')

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['Train', 'Module', 'Energy', 'Waiting', 'InVehicle'])#, rotation=45)
    ax.set_ylabel('Cost (1000$/hr)')
    ax.set_yticks(np.arange(0, 1.61e5, 0.2e5))
    ax.set_yticklabels(np.arange(0, 161, 20))
    ax.grid()

    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/figure6-indiv_barplot.png')
    plt.show()


def plot_v_assessment():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, v_ in enumerate(np.arange(100, 405, 5)):
        tmp_ys_nst = []
        tmp_xs_nst = []
        tmp_ys_ktx = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        for j in range(1, 101):
            x_nst = min_x_NST(params, v_/3.6, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v_/3.6, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v_/3.6, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v_/3.6, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v_/3.6, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.bar(np.arange(100, 405, 5), (
            np.stack(J_ktx)[:, 0] - np.stack(J_nst)[:, 0]) / np.stack(J_ktx)[:, 0],
            width=4, alpha=0.2, color='red', label='Relative savings (%)')
    ax2.set_ylabel('reduction rate (%)')
    ax2.set_ylim([0, 0.30])
    # ax2.set_yticks(np.arange(0, 0.4 + 0.05, 0.1))

    ax1.plot(np.arange(100, 405, 5), np.stack(J_ktx)[:, 0], label='AST')
    ax1.plot(np.arange(100, 405, 5), np.stack(J_nst)[:, 0], label='NST', color='tab:orange')
    ax1.plot([50, 300], [np.stack(J_ktx)[40, 0],np.stack(J_ktx)[40, 0]], color='grey', linestyle='dashed')

    min_speed_idx = np.min(np.where(np.stack(J_nst)[:, 0] < np.stack(J_ktx)[40, 0])[0])
    ax1.plot([100 + 5 * min_speed_idx, 100 + 5 * min_speed_idx],
             [np.stack(J_nst)[min_speed_idx, 0], 0], color='grey', linestyle='dashed')
    ax1.plot([300, 300], [np.stack(J_ktx)[40, 0], 0], color='grey', linestyle='dashed')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # fig.suptitle(r'Optimal cost by different $v_c$')
    ax1.set_ylabel('generalized cost (1,000 $/hr)')
    ax1.set_xlabel(r'$v_c$ (km/h)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    # ax1.legend()
    ax1.set_ylim(140000, 410000)
    ax1.set_xlim(85, 415)

    ax1.set_yticks(np.arange(1.5e5, 4.5e5, 5e4))
    ax1.set_yticklabels(np.arange(150, 450, 50))
    ax1.set_xticks([100, 150, 200, 230, 250, 300, 350, 400])
    # ax1.set_xticklabels([100, 150, 200, '230\n' + r'$min(J^{1})<min(J^{0})$', 250, 300, 350, 400])
    ax1.set_xticklabels([100, 150, 200, 230, 250, 300, 350, 400])

    for yy in np.arange(1.5e5, 4.5e5, 5e4):
        ax1.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(100, 401, 50):
        ax1.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    plt.tight_layout()
    plt.savefig('figures/figure11-v_assessment.png')
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.bar(np.arange(100, 410, 5), (
            np.stack(J_ktx)[:, 0] - np.stack(J_nst)[:, 0]) / np.stack(J_ktx)[:, 0],
            width=4, alpha=0.2 , color='red')
    ax2.set_ylabel('reduction rate (%)')
    ax2.set_ylim([0, 0.4])
    ax2.set_yticks(np.arange(0, 0.4 + 0.05, 0.1))

    ax1.plot(np.arange(100, 410, 5), np.stack(J_ktx)[:, 0] - np.stack(J_nst)[:, 0])
    fig.suptitle(r'Cycle time difference')
    ax1.set_ylabel('cost (1,000 $/hr)')
    ax1.set_xlabel(r'$v_c$ (km/h)')
    ax1.set_yticks(np.arange(2.5e4, 6.5e4 + 1, 1e4))
    ax1.set_yticklabels(np.arange(25, 65 + 1, 10))

    plt.tight_layout()
    plt.savefig('figures/v_difference.png')
    plt.show()


def plot_invehicle_by_v():
    IVTT_KTX = []
    IVTT_NST = []
    demand = np.sum(OD_in) + np.sum(OD_out)
    for i, v_ in enumerate(np.arange(100, 410, 5)):
        _, ivtt_ktx = passenger_travel_times_KTX(params, v_ / 3.6, y, stations, OD_out, OD_in)
        _, ivtt_nst = passenger_travel_times_NST(params, v_ / 3.6, y, stations, OD_out, OD_in)


        IVTT_KTX.append(ivtt_ktx / demand)
        IVTT_NST.append(ivtt_nst / demand)

    IVTT_KTX = np.array(IVTT_KTX)
    IVTT_NST = np.array(IVTT_NST)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax2.bar(np.arange(100, 410, 5), (IVTT_KTX - IVTT_NST) / IVTT_KTX,
            width=4, alpha=0.2, color='red')
    ax2.set_ylabel('reduction rate (%)')
    ax2.set_ylim([0, 0.55])
    ax2.set_yticks(np.arange(0, 0.5 + 0.05, 0.1))

    ax1.plot(np.arange(100, 410, 5), IVTT_KTX * 60, label='KTX')
    ax1.plot(np.arange(100, 410, 5), IVTT_NST * 60, label='NST')
    ax1.legend()
    ax1.set_ylabel('IVTT per passenger (min)')
    ax1.set_xlabel('min')
    ax1.set_yticks(np.arange(40, 162, 30))
    # ax1.set_yticklabels(np.arange(25, 65 + 1, 10))
    plt.show()


def plot_energy_cost():
    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(2, 11):
        params['N_dc'] = i
        tmp_z = np.ones(params['n_station'] * 2) * params['N_dc']
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=tmp_z, regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, regen=True, aux=True))

    params['N_dc'] = 4
    tmp_z = np.ones(params['n_station'] * 2) * params['N_dc']

    plt.plot(np.arange(2, 11), NST_energy_consumptions)
    plt.plot(np.arange(2, 11), KTX_energy_consumptions)
    # plt.plot([4, 4], [15000, 50000], '--', c='grey')
    plt.legend(['NST', 'KTX'])
    plt.xlabel('Number of decoupled units (units)')
    plt.title('N_dc')
    plt.show()

    Z = np.ones(params['n_station'] * 2) * 4

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(15000, 75000, 1000):
        params['s'] = i * 1.0
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, mode='ss', regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, mode='ss', regen=True, aux=True))
        # print('s: {} / E_dc: {:.3f} / E_da: {:.3f} / E_dd: {:.3f} / total: {:.3f}'.format(
        #     i, E_dc(coefficients, params, v, N, mode='ss'), E_da(coefficients, params, v, N, mode='ss'), E_dd(coefficients, params, v, N, mode='ss'),
        #     E_dc(coefficients, params, v, N, mode='ss') + E_da(coefficients, params, v, N, mode='ss') - E_dd(coefficients, params, v, N, mode='ss')))

    params['s'] = 46.4 * 1000

    plt.plot(np.arange(15, 75), NST_energy_consumptions)
    plt.plot(np.arange(15, 75), KTX_energy_consumptions)
    plt.legend(['NST', 'KTX'])
    plt.xlabel('Stop spacing (km)')
    plt.ylabel('Energy (kWh)')
    plt.title('NST vs. KTX Energy consumption - s')
    # plt.show()
    # plt.savefig('Comparison_spacing.png')

    tmp_z = np.ones(params['n_station'] * 2) * params['N_dc']

    figs, axs = plt.subplots(2, 3, figsize=(12, 6))
    NST_energy_consumptions = [energy_NST(coefficients, params, v_/3.6, N, Z=tmp_z, regen=True, aux=True) for v_ in range(100, 401, 10)]
    KTX_energy_consumptions = [energy_KTX(coefficients, params, v_/3.6, N, regen=True, aux=True) for v_ in range(100, 401, 10)]

    figs.suptitle('NST vs. KTX Energy Consumption')
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
        tmp_z = np.ones(params['n_station'] * 2) * params['N_dc']
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=tmp_z, regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, regen=True, aux=True))

    params['N_dc'] = 4
    tmp_z = np.ones(params['n_station'] * 2) * params['N_dc']

    axs[0, 1].plot(np.arange(2, 11), NST_energy_consumptions)
    axs[0, 1].plot(np.arange(2, 11), KTX_energy_consumptions)
    axs[0, 1].plot([4, 4], [15000, 50000], '--', c='grey')
    axs[0, 1].legend(['NST', 'KTX'])
    axs[0, 1].set_xlabel('Number of decoupled units (units)')
    axs[0, 1].set_title('N_dc')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i in range(15000, 75000, 1000):
        params['s'] = i * 1.0
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=tmp_z, regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, regen=True, aux=True))
    params['s'] = 46.4 * 1000

    axs[0, 2].plot(np.arange(15, 75), NST_energy_consumptions)
    axs[0, 2].plot(np.arange(15, 75), KTX_energy_consumptions)
    axs[0, 2].plot([46.4, 46.4], [15000, 50000], '--', c='grey')
    axs[0, 2].legend(['NST', 'KTX'])
    axs[0, 2].set_xlabel('Stop spacing (km)')
    axs[0, 2].set_title('s')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, a_ in enumerate(np.arange(0.3, 1.55, 0.05)):
        params['a'] = a_
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=tmp_z, regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, regen=True, aux=True))
    params['a'] = 0.68

    axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), NST_energy_consumptions)
    axs[1, 0].plot(np.arange(0.3, 1.55, 0.05), KTX_energy_consumptions)
    axs[1, 0].plot([0.68, 0.68], [15000, 50000], '--', c='grey')
    axs[1, 0].legend(['NST', 'KTX'])
    axs[1, 0].set_xlabel('acceleration (m/s^2)')
    axs[1, 0].set_title('a')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    for i, e_ in enumerate(np.arange(25, 80, 2.5)):
        params['e_sub'] = e_ * 1000.0
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=tmp_z, regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, regen=True, aux=True))
    params['e_sub'] = 52.5 * 1000

    axs[1, 1].plot(np.arange(25, 80, 2.5), NST_energy_consumptions)
    axs[1, 1].plot(np.arange(25, 80, 2.5), KTX_energy_consumptions)
    axs[1, 1].plot([52.5, 52.5], [15000, 50000], '--', c='grey')
    axs[1, 1].legend(['NST', 'KTX'])
    axs[1, 1].set_xlabel('e_sub (kW)')
    axs[1, 1].set_title('e_sub')

    NST_energy_consumptions = []
    KTX_energy_consumptions = []
    oe = params['eta_reg']
    for i, e_ in enumerate(np.arange(0.1, 0.9, 0.1)):
        params['eta_reg'] = e_
        NST_energy_consumptions.append(energy_NST(coefficients, params, v, N, Z=tmp_z, regen=True, aux=True))
        KTX_energy_consumptions.append(energy_KTX(coefficients, params, v, N, regen=True, aux=True))
    params['eta_reg'] = oe
    print(params['eta_reg'])

    axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), NST_energy_consumptions)
    axs[1, 2].plot(np.arange(0.1, 0.9, 0.1), KTX_energy_consumptions)
    axs[1, 2].plot([0.2, 0.2], [15000, 50000], '--', c='grey')
    axs[1, 2].legend(['NST', 'KTX'])
    axs[1, 2].set_xlabel('eta_reg')
    axs[1, 2].set_title('eta_reg')

    axs[0, 0].set_ylim([28000, 40000])
    axs[0, 1].set_ylim([28000, 40000])
    axs[0, 2].set_ylim([28000, 40000])
    axs[1, 0].set_ylim([28000, 40000])
    axs[1, 1].set_ylim([28000, 40000])
    axs[1, 2].set_ylim([28000, 40000])

    figs.supxlabel('(Grey dashed lines indicate the default cases)', fontsize=10)

    plt.tight_layout()
    plt.show()
    # plt.savefig('figures/Comparison_vc_Ndc_s_scaled_v2.png')

    ##
    stations = [0.0, 14.9, 32.1, 103.9, 131.7, 165.1, 236.5, 277.9, 291.3, 338.2, 365.7, 415.0]


def plot_mu_0_sensitivity():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(1.0, 10, 19)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu01'] = m_ * params['mu00']
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    plt.plot(np.linspace(1.0, 10, 19), np.stack(J_ktx)[:, 0], label='AST')
    plt.plot(np.linspace(1.0, 10, 19), np.stack(J_nst)[:, 0], label='NST')
    plt.plot([1.5, 1.5], [0, np.stack(J_ktx)[1, 0]], '--', c='grey')
    # plt.title(r'Optimal cost by different $m_0^1/m_0^0$')
    plt.ylabel('generalized cost (1,000 $/hr)')
    plt.xlabel(r'Relative train cost ratios ($\mu_0^1/\mu_0^0$)')
    ax = plt.gca()
    ax.set_yticks(np.arange(175000, 400001, 25000))
    ax.set_yticklabels(np.arange(175, 401, 25))
    plt.ylim([160000, 280000])
    ax.set_xticks(np.concatenate(([1, 1.5], np.arange(3, 10, 2), [10])))
    plt.legend()
    for yy in np.arange(175e3, 276e3, 25e3):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(1, 10, 2):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figures13a-mu0100_sensitivity.png')
    plt.show()


    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(params['mu00'] / 4, params['mu00'] * 3, 12)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu00'] = m_
        # params['mu41'] = 2.0 * params['mu40']
        params['mu01'] = m_ * 1.5
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])
    params['mu00'] = (labor_price + roll_train_pr)
    params['mu01'] = params['mu00'] * 1.5

    plt.plot(np.linspace(params['mu00'] / 4, params['mu00'] * 3, 12),
             np.stack(J_ktx)[:, 0], label='AST')
    plt.plot(np.linspace(params['mu00'] / 4, params['mu00'] * 3, 12),
             np.stack(J_nst)[:, 0], label='NST')
    plt.plot([params['mu00'], params['mu00']], [0, np.stack(J_ktx)[3, 0]], '--', c='grey')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # plt.title('Sensitivity of  cost by different m0')
    plt.ylabel('generalized cost (1,000 $/hr)')
    # plt.xlabel(r'$\mathregular{\mu_0^0}$'.join(' (default = 112.1USD/hr)'))
    plt.xlabel(r'train cost coefficient $\mathregular{\mu_0^0}$ (\$/hr)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 450000, 50000))
    ax.set_yticklabels(np.arange(50, 450, 50))
    ax.set_xticks(np.concatenate(([50, np.round(params['mu00'], 1)],
                                  np.arange(150, 351, 50))))

    plt.ylim([30000, 430000])
    plt.legend()
    for yy in np.arange(50e3, 401e3, 50e3):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(50, 351, 50):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    plt.tight_layout()
    plt.savefig('figures/figures14a-mu0_sensitivity.png')
    plt.show()

    # ols_x = np.arange(20, 400, 10)
    # ols_x = sm.add_constant(ols_x)
    # model_nst = sm.OLS(np.stack(J_nst)[:, 0], ols_x)
    # results_nst = model_nst.fit()
    # model_ktx = sm.OLS(np.stack(J_ktx)[:, 0], ols_x)
    # results_ktx = model_ktx.fit()
    # print(results_ktx.params)
    # print(results_nst.params)
    # plt.plot(np.arange(20, 400, 10), np.stack(J_nst)[:, 0], '.', label='NST', color='tab:blue')
    # plt.plot(np.arange(20, 400, 10), np.stack(J_ktx)[:, 0], '.', label='KTX', color='tab:orange')
    # plt.plot(np.arange(20, 400, 10), results_nst.predict(ols_x), color='tab:blue')
    # plt.plot(np.arange(20, 400, 10), results_ktx.predict(ols_x), color='tab:orange')
    # # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # plt.title('Optimal cost by different m0')
    # plt.ylabel('cost (1,000 $/hr)')
    # plt.xlabel('m0 ($/unit-hr)')
    # ax = plt.gca()
    # ax.set_yticks(np.arange(50000, 350000, 50000))
    # ax.set_yticklabels(np.arange(50, 350, 50))
    # plt.ylim([30000, 330000])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('figures/mu0_sensitivity.png')
    # plt.show()
    #
    # params['mu00'] = (labor_price + roll_train_pr)
    # params['mu01'] = params['mu00'] * 1.5


def plot_mu_1_sensitivity():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(1.0, 10, 19)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu11'] = m_ * params['mu10']
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    params['mu10'] = (module_price + infra_price + roll_unit_pr)
    params['mu11'] = params['mu10'] * 2

    idx = np.min(np.where(np.stack(J_ktx)[:, 0] < np.stack(J_nst)[:, 0])[0])
    val = np.linspace(1, 10, 19)[idx]
    plt.plot(np.linspace(1.0, 10, 19), np.stack(J_ktx)[:, 0], label='AST')
    plt.plot(np.linspace(1.0, 10, 19), np.stack(J_nst)[:, 0], label='NST')
    # plt.plot([val, val], [0, np.stack(J_ktx)[idx, 0]], c='black')
    plt.plot([2, 2], [0, np.stack(J_ktx)[2, 0]], '--', c='grey')
    plt.ylabel('generalized cost (1,000 $/hr)')
    plt.xlabel(r'Relative unit cost ratios ($\mu_1^1/\mu_1^0$)')
    ax = plt.gca()
    ax.set_yticks(np.arange(175000, 400001, 25000))
    ax.set_yticklabels(np.arange(175, 401, 25))
    plt.ylim([160000, 280000])
    ax.set_xticks(np.concatenate(([1, 2], np.arange(3, 10, 2), [10])))

    plt.legend()
    for yy in np.arange(175e3, 276e3, 25e3):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(1, 10, 2):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure13b-mu1110_sensitivity.png')
    plt.show()


    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(params['mu10'] / 4, params['mu10'] * 3, 12)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu10'] = m_
        params['mu11'] = 2.0 * params['mu10']
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    params['mu10'] = (module_price + infra_price + roll_unit_pr)
    params['mu11'] = params['mu10'] * 2

    plt.plot(np.linspace(params['mu10'] / 4, params['mu10'] * 3, 12),
             np.stack(J_ktx)[:, 0], label='AST')
    plt.plot(np.linspace(params['mu10'] / 4, params['mu10'] * 3, 12),
             np.stack(J_nst)[:, 0], label='NST')
    plt.plot([params['mu10'], params['mu10']], [0, np.stack(J_ktx)[4, 0]], '--', c='grey')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # plt.title('Optimal cost by different m10')
    plt.ylabel('generalized cost (1,000 $/hr)')
    plt.xlabel(r'unit cost coefficient $\mathregular{\mu_1^0}$ (\$/unit-hr)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 450000, 50000))
    ax.set_yticklabels(np.arange(50, 450, 50))
    plt.ylim([30000, 430000])
    ax.set_xticks(np.concatenate(([20, 40, np.round(params['mu10'])], np.arange(60, 141, 20))))

    plt.legend()
    for yy in np.arange(50e3, 401e3, 50e3):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(20, 141, 20):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure14b-mu1_sensitivity.png')
    plt.show()

    ols_x = np.arange(2, 42, 2)
    ols_x = sm.add_constant(ols_x)
    model_nst = sm.OLS(np.stack(J_nst)[:, 0], ols_x)
    results_nst = model_nst.fit()
    model_ktx = sm.OLS(np.stack(J_ktx)[:, 0], ols_x)
    results_ktx = model_ktx.fit()
    print(results_ktx.params)
    print(results_nst.params)
    plt.plot(np.arange(2, 42, 2), np.stack(J_nst)[:, 0], '.', label='NST', color='tab:blue')
    plt.plot(np.arange(2, 42, 2), np.stack(J_ktx)[:, 0], '.', label='KTX', color='tab:orange')
    plt.plot(np.arange(2, 42, 2), results_nst.predict(ols_x), color='tab:blue')
    plt.plot(np.arange(2, 42, 2), results_ktx.predict(ols_x), color='tab:orange')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    plt.title('Optimal cost by different m10')
    plt.ylabel('cost (1,000 $/hr)')
    plt.xlabel('m10 ($/unit-hr)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 350000, 50000))
    ax.set_yticklabels(np.arange(50, 350, 50))
    plt.ylim([30000, 330000])
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/mu1_sensitivity.png')
    plt.show()

    params['mu10'] = (module_price + infra_price + roll_unit_pr)
    params['mu11'] = params['mu10'] * 2


def plot_mu_2_sensitivity():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(params['mu2'] / 4, params['mu2'] * 3, 12)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu2'] = m_
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])
    params['mu2'] = 0.14
    plt.plot(np.linspace(params['mu2'] / 4, params['mu2'] * 3, 12),
             np.stack(J_ktx)[:, 0], label='AST')
    plt.plot(np.linspace(params['mu2'] / 4, params['mu2'] * 3, 12),
             np.stack(J_nst)[:, 0], label='NST')
    plt.plot([params['mu2'], params['mu2']], [0, np.stack(J_ktx)[3, 0]], '--', c='grey')
    # plt.title('Optimal cost by different m2')
    plt.ylabel('generalized cost ($/hr)')
    plt.xlabel(r'electricity price $\mathregular{\mu_2}$ (\$/kWh)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 450000, 50000))
    ax.set_yticklabels(np.arange(50, 450, 50))
    plt.ylim([30000, 430000])

    ax.set_xticks(np.concatenate(([0.05, 0.10, np.round(params['mu2'], 2)],
                                  np.arange(0.20, 0.41, 0.05))))
    plt.legend()
    for yy in np.arange(50e3, 401e3, 50e3):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0.05, 0.41, 0.05):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure14c-mu2_sensitivity.png')
    plt.show()

    ols_x = np.arange(0.05, 0.30, 0.01)
    ols_x = sm.add_constant(ols_x)
    model_nst = sm.OLS(np.stack(J_nst)[:, 0], ols_x)
    results_nst = model_nst.fit()
    model_ktx = sm.OLS(np.stack(J_ktx)[:, 0], ols_x)
    results_ktx = model_ktx.fit()
    print(results_nst.params)
    print(results_ktx.params)
    plt.plot(np.arange(0.05, 0.30, 0.01), np.stack(J_nst)[:, 0], '.', label='NST', color='tab:blue')
    plt.plot(np.arange(0.05, 0.30, 0.01), np.stack(J_ktx)[:, 0], '.', label='KTX', color='tab:orange')
    plt.plot(np.arange(0.05, 0.30, 0.01), results_nst.predict(ols_x), color='tab:blue')
    plt.plot(np.arange(0.05, 0.30, 0.01), results_ktx.predict(ols_x), color='tab:orange')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    plt.title('Optimal cost by different m2')
    plt.ylabel('cost ($/hr)')
    plt.xlabel('m2 ($/kWh)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 350000, 50000))
    ax.set_yticklabels(np.arange(50, 350, 50))
    plt.ylim([30000, 330000])
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/mu2_sensitivity.png')
    plt.show()

    params['mu2'] = 0.14
    # print(xs_nst)
    # print(ys_nst)


def plot_mu_3_sensitivity():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(params['mu3'] / 4, params['mu3'] * 3, 12)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu3'] = m_
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    params['mu3'] = 20.25

    plt.plot(np.linspace(params['mu3'] / 4, params['mu3'] * 3, 12),
             np.stack(J_ktx)[:, 0], label='AST')
    plt.plot(np.linspace(params['mu3'] / 4, params['mu3'] * 3, 12),
             np.stack(J_nst)[:, 0], label='NST')
    plt.plot([params['mu3'], params['mu3']], [0, np.stack(J_ktx)[3, 0]], '--', c='grey')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # plt.title('Optimal cost by different m3')
    plt.ylabel('generalized cost (1,000 $/hr)')
    plt.xlabel(r'value of travel time $\mathregular{\mu_3}$ (\$/px-hr)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 450000, 50000))
    ax.set_yticklabels(np.arange(50, 450, 50))
    plt.ylim([30000, 430000])
    ax.set_xticks(np.concatenate(([10, np.round(params['mu3'])], np.arange(30, 61, 10))))
    plt.legend()
    for yy in np.arange(50e3, 401e3, 50e3):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(10, 61, 10):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure14d-mu3_sensitivity.png')
    plt.show()

    ols_x = np.arange(2.0, 32, 2)
    ols_x = sm.add_constant(ols_x)
    model_nst = sm.OLS(np.stack(J_nst)[:, 0], ols_x)
    results_nst = model_nst.fit()
    model_ktx = sm.OLS(np.stack(J_ktx)[:, 0], ols_x)
    results_ktx = model_ktx.fit()
    print(results_ktx.params)
    print(results_nst.params)
    plt.plot(np.arange(2.0, 32, 2), np.stack(J_nst)[:, 0], '.', label='NST', color='tab:blue')
    plt.plot(np.arange(2.0, 32, 2), np.stack(J_ktx)[:, 0], '.', label='KTX', color='tab:orange')
    plt.plot(np.arange(2.0, 32, 2), results_nst.predict(ols_x), color='tab:blue')
    plt.plot(np.arange(2.0, 32, 2), results_ktx.predict(ols_x), color='tab:orange')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    plt.title('Optimal cost by different m3')
    plt.ylabel('cost (1,000 $/hr)')
    plt.xlabel('m3 ($/px-hr)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 350000, 50000))
    ax.set_yticklabels(np.arange(50, 350, 50))
    plt.ylim([30000, 330000])
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/mu3_sensitivity.png')
    plt.show()

    params['mu3'] = 20.25
    print(xs_nst)
    print(ys_nst)


def pareto_search():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []

    search_space = np.arange(0, 1.001, 0.01)
    for i, w_ in enumerate(search_space):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        for j in range(1, 101):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station',
                                          pareto=1, w1=w_)
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station',
                                          pareto=1, w1=w_)
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    plt.plot(search_space, np.stack(J_nst)[:, 0], label='NST')
    plt.plot(search_space, np.stack(J_ktx)[:, 0], label='AST')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    plt.title('w * agency_cost + (1-w) * user_cost')
    plt.ylabel('cost ($/hr)')
    plt.xlabel('w')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('figures/mu1_sensitivity.png')
    plt.show()

    agency_cost_NST = np.sum(np.stack(J_nst)[:, 1:3], axis=1)
    user_cost_NST = np.sum(np.stack(J_nst)[:, 3:], axis=1)
    t_units_nst = np.array(xs_nst) * np.array(ys_nst) + np.array([np.sum(z_) for z_ in zs])
    # for i in range(0, 30):
    #     plt.plot([-15000, 30000 + 15000 * i], [30000 + 15000 * i, -15000],
    #              c='grey', alpha=0.4)
    plt.plot(agency_cost_NST, user_cost_NST, c='tab:orange')
    plt.scatter(agency_cost_NST, user_cost_NST,
                c=t_units_nst, cmap='viridis', vmin=160, vmax=300, zorder=-1)
    plt.colorbar(label="Total number of units")
    plt.ylim([130000, 400000])
    plt.xlim([20000, 55000])
    # cbar.set_ticks([1, 20, 40, 60, 80, 100])
    # cbar.set_ticklabels(["1", "20", "40", "60", "80", ">100"])

    plt.ylabel('user cost ($/hr)')
    plt.xlabel('agency cost ($/hr)')
    plt.title('pareto search - NST')
    plt.tight_layout()
    plt.show()

    agency_cost_KTX = np.sum(np.stack(J_ktx)[:, 1:3], axis=1)
    user_cost_KTX = np.sum(np.stack(J_ktx)[:, 3:], axis=1)
    # for i in range(0, 30):
    #     plt.plot([-15000, 30000 + 15000 * i], [30000 + 15000 * i, -15000],
    #              c='grey', alpha=0.4)
    t_units_ktx = np.array(xs_ktx) * np.array(ys_ktx)
    plt.plot(agency_cost_KTX, user_cost_KTX, c='tab:blue')
    plt.scatter(agency_cost_KTX, user_cost_KTX,
                c=t_units_ktx, cmap='viridis', vmin=160, vmax=300, zorder=-1)
    plt.ylim([130000, 400000])
    plt.xlim([20000, 55000])
    plt.colorbar(label="Total number of units")
    # cbar.set_ticks([1, 20, 40, 60, 80, 100])
    # cbar.set_ticklabels(["1", "20", "40", "60", "80", ">100"])

    plt.ylabel('user cost ($/hr)')
    plt.xlabel('agency cost ($/hr)')
    plt.title('pareto search - AST')
    plt.tight_layout()
    plt.show()

    ######
    agency_cost_KTX = np.sum(np.stack(J_ktx)[:, [1, 2, 5]], axis=1)
    user_cost_KTX = np.sum(np.stack(J_ktx)[:, 3:5], axis=1)
    t_units_ktx = np.array(xs_ktx) * np.array(ys_ktx)
    plt.plot(agency_cost_KTX, user_cost_KTX, c='tab:blue', label='AST')
    plt.scatter(agency_cost_KTX, user_cost_KTX,
                c=np.stack(J_ktx)[:, 0], cmap='viridis', zorder=-1)

    agency_cost_NST = np.sum(np.stack(J_nst)[:, [1, 2, 5]], axis=1)
    user_cost_NST = np.sum(np.stack(J_nst)[:, 3:5], axis=1)
    t_units_nst = np.array(xs_nst) * np.array(ys_nst) + np.array([np.sum(z_) for z_ in zs])
    # for i in range(0, 30):
    #     plt.plot([-15000, 30000 + 15000 * i], [30000 + 15000 * i, -15000],
    #              c='grey', alpha=0.4)
    plt.plot(agency_cost_NST, user_cost_NST, c='tab:orange', label='NST')
    plt.scatter(agency_cost_NST, user_cost_NST,
                c=np.stack(J_nst)[:, 0], cmap='viridis', zorder=-1)
    cbar = plt.colorbar(label="Total social cost (1,000$/hr)")
    plt.ylim([100000, 400000])
    plt.xlim([23000, 89000])
    ax = plt.gca()
    ax.set_yticks(np.arange(100000, 400001, 50000))
    ax.set_yticklabels(np.arange(100, 401, 50))
    ax.set_xticks(np.arange(30000, 80001, 10000))
    ax.set_xticklabels(np.arange(30, 81, 10))
    cbar.set_ticks(np.arange(40000, 120000+1, 20000))
    cbar.set_ticklabels(np.arange(40, 120+1, 20))
    plt.legend()

    plt.ylabel('user cost (1,000$/hr)')
    plt.xlabel('agency cost (1,000$/hr)')
    # plt.title('pareto search')
    plt.tight_layout()
    plt.show()

    ############
    agency_cost_KTX = np.sum(np.stack(J_ktx)[:, [1, 2, 5]], axis=1)
    user_cost_KTX = np.sum(np.stack(J_ktx)[:, 3:5], axis=1)
    t_units_ktx = np.array(xs_ktx) * np.array(ys_ktx)
    plt.plot(agency_cost_KTX, user_cost_KTX, c='tab:blue', label='AST')
    plt.scatter(agency_cost_KTX, user_cost_KTX,
                c=t_units_ktx, vmin=160, vmax=300, cmap='viridis', zorder=-1)

    agency_cost_NST = np.sum(np.stack(J_nst)[:, [1, 2, 5]], axis=1)
    user_cost_NST = np.sum(np.stack(J_nst)[:, 3:5], axis=1)
    t_units_nst = np.array(xs_nst) * np.array(ys_nst) + np.array([np.sum(z_) for z_ in zs])
    # for i in range(0, 30):
    #     plt.plot([-15000, 30000 + 15000 * i], [30000 + 15000 * i, -15000],
    #              c='grey', alpha=0.4)
    plt.plot(agency_cost_NST, user_cost_NST, c='tab:orange', label='NST')
    plt.scatter(agency_cost_NST, user_cost_NST,
                c=t_units_nst, vmin=160, vmax=300, cmap='viridis', zorder=-1)
    cbar = plt.colorbar(label="Number of total units")
    plt.ylim([100000, 400000])
    plt.xlim([23000, 89000])
    ax = plt.gca()
    ax.set_yticks(np.arange(100000, 400001, 50000))
    ax.set_yticklabels(np.arange(100, 401, 50))
    ax.set_xticks(np.arange(30000, 80001, 10000))
    ax.set_xticklabels(np.arange(30, 81, 10))
    # cbar.set_ticks(np.arange(40000, 120000+1, 20000))
    # cbar.set_ticklabels(np.arange(40, 120+1, 20))
    plt.legend()

    plt.ylabel('user cost (1,000$/hr)')
    plt.xlabel('agency cost (1,000$/hr)')
    # plt.title('pareto search')
    plt.tight_layout()
    plt.show()

    fs =14
    # Agency cost

    tmp_ratio = np.abs(agency_cost_KTX - agency_cost_NST) / agency_cost_KTX

    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    ax1.plot(np.arange(0, 1.001, 0.01), agency_cost_KTX, c='tab:blue', label='AST')
    ax1.plot(np.arange(0, 1.001, 0.01), agency_cost_NST, c='tab:orange', label='NST')
    ax1.set_ylabel('operator cost (1,000$/hr)', fontsize=fs+2)
    ax1.set_ylim(2e4, 5e5)
    ax1.set_yticks(np.arange(5e4, 5.2e5, 1e5))
    ax1.set_yticklabels(np.arange(50, 520, 100), fontsize=fs)
    ax1.set_xlabel(r'Weight factor $\omega$', fontsize=fs+2)

    # ax2.bar(np.arange(0, 1.001, 0.01), tmp_ratio, color='blue', label='Cost advantage', width=0.01, alpha=0.2)
    # ax2.set_ylabel('Cost ratio', fontsize=fs+2)
    # ax2.set_ylim(0.0, 0.85)
    # ax2.set_yticks(np.arange(0, 0.9, 0.2))
    # ax2.tick_params(axis='both', labelsize=fs)
    ax1.tick_params(axis='both', labelsize=fs)
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=fs+2)
    ax1.legend(fontsize=fs+2)

    for yy in np.arange(50e3, 451e3, 100e3):
        ax1.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0.0, 1.1, 0.2):
        ax1.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    plt.tight_layout()
    plt.savefig('figures/figure12a-omega_sensitivity_operator.png')
    plt.show()

    # User cost
    tmp_ratio = np.abs(user_cost_KTX - user_cost_NST) / user_cost_KTX

    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    ax1.plot(np.arange(0, 1.001, 0.01), user_cost_KTX, c='tab:blue', label='AST')
    ax1.plot(np.arange(0, 1.001, 0.01), user_cost_NST, c='tab:orange', label='NST')
    ax1.set_ylabel('user cost (1,000$/hr)', fontsize=fs+2)
    ax1.set_ylim(2e4, 5e5)
    ax1.set_yticks(np.arange(5e4, 5.2e5, 1e5))
    ax1.set_yticklabels(np.arange(50, 520, 100), fontsize=fs)
    ax1.set_xlabel(r'Weight factor $\omega$', fontsize=fs+2)

    # ax2.bar(np.arange(0, 1.001, 0.01), tmp_ratio, color='red', label='Cost advantage', width=0.01, alpha=0.2)
    # ax2.set_ylabel('Cost ratio', fontsize=fs+2)
    # ax2.set_ylim(0.0, 0.85)
    # ax2.set_yticks(np.arange(0, 0.9, 0.2))
    # ax2.tick_params(axis='both', labelsize=fs)
    ax1.tick_params(axis='both', labelsize=fs)
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=fs+2)
    ax1.legend(fontsize=fs + 2)

    for yy in np.arange(50e3, 451e3, 100e3):
        ax1.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0.0, 1.1, 0.2):
        ax1.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    plt.tight_layout()
    plt.savefig('figures/figure12b-omega_sensitivity_user.png')
    plt.show()

    # Total cost
    all_cost_KTX = np.sum(np.stack(J_ktx)[:, 1:], axis=1)
    all_cost_NST = np.sum(np.stack(J_nst)[:, 1:], axis=1)
    tmp_ratio = (all_cost_KTX - all_cost_NST) / all_cost_KTX

    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    ax1.plot(np.arange(0, 1.001, 0.01), all_cost_KTX, c='tab:blue', label='AST')
    ax1.plot(np.arange(0, 1.001, 0.01), all_cost_NST, c='tab:orange', label='NST')
    ax1.set_ylabel('generalized cost (1,000$/hr)', fontsize=fs+2)
    ax1.set_ylim(2e4, 5e5)
    ax1.set_yticks(np.arange(5e4, 5.2e5, 1e5))
    ax1.set_yticklabels(np.arange(50, 520, 100), fontsize=fs)
    ax1.set_xlabel(r'Weight factor $\omega$', fontsize=fs+2)

    # ax2.bar(np.arange(0, 1.001, 0.01), tmp_ratio, color='red', label='Cost advantage', width=0.01, alpha=0.2)
    # ax2.set_ylabel('Cost ratio', fontsize=fs+2)
    # ax2.set_ylim(0.0, 0.85)
    # ax2.set_yticks(np.arange(0, 0.9, 0.2))
    # ax2.tick_params(axis='both', labelsize=fs)
    ax1.tick_params(axis='both', labelsize=fs)
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=fs+2)
    ax1.legend(fontsize=fs + 2)

    for yy in np.arange(50e3, 451e3, 100e3):
        ax1.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0.0, 1.1, 0.2):
        ax1.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    plt.tight_layout()
    plt.savefig('figures/figure12c-omega_sensitivity_generalized.png')
    plt.show()

    # Total cost
    all_cost_KTX = np.stack(J_ktx)[:, 0]
    all_cost_NST = np.stack(J_nst)[:, 0]
    tmp_ratio = (all_cost_KTX - all_cost_NST) / all_cost_KTX
    tmp_ratio2 = (all_cost_NST - all_cost_KTX) / all_cost_KTX

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(0, 1.001, 0.01), all_cost_KTX, c='tab:blue', label='AST')
    ax1.plot(np.arange(0, 1.001, 0.01), all_cost_NST, c='tab:orange', label='NST')
    ax1.set_ylabel('Total cost (1,000$/hr)', fontsize=fs+2)
    ax1.set_ylim(2e4, 5e5)
    ax1.set_yticks(np.arange(5e4, 5.2e5, 1e5))
    ax1.set_yticklabels(np.arange(50, 520, 100), fontsize=fs)
    ax1.set_xlabel(r'Weight factor $\omega$', fontsize=fs+2)

    ax2.bar(np.arange(0, 1.001, 0.01), tmp_ratio, color='red', label='Cost advantage', width=0.01, alpha=0.2)
    ax2.bar(np.arange(0, 1.001, 0.01), tmp_ratio2, color='blue', label='Cost advantage', width=0.01, alpha=0.2)
    ax2.set_ylabel('Cost ratio', fontsize=fs+2)
    ax2.set_ylim(0.0, 0.85)
    ax2.set_yticks(np.arange(0, 0.9, 0.2))
    ax2.tick_params(axis='both', labelsize=fs)
    ax1.tick_params(axis='both', labelsize=fs)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=fs+2)
    plt.tight_layout()
    plt.show()


def pareto_2d():
    GREEN = "\033[32m"
    RESET = "\033[0m"

    search_space = np.arange(0, 10.1, 0.1)

    J_nst = np.zeros((len(search_space), len(search_space)))
    module_nst = np.zeros((len(search_space), len(search_space)))
    electricity_nst = np.zeros((len(search_space), len(search_space)))
    waiting_nst = np.zeros((len(search_space), len(search_space)))
    travel_nst = np.zeros((len(search_space), len(search_space)))
    J_ktx = np.zeros((len(search_space), len(search_space)))
    module_ktx = np.zeros((len(search_space), len(search_space)))
    electricity_ktx = np.zeros((len(search_space), len(search_space)))
    waiting_ktx = np.zeros((len(search_space), len(search_space)))
    travel_ktx = np.zeros((len(search_space), len(search_space)))

    ys_nst = np.zeros((len(search_space), len(search_space)))
    xs_nst = np.zeros((len(search_space), len(search_space)))
    zs = np.zeros((len(search_space), len(search_space)))

    ys_ktx = np.zeros((len(search_space), len(search_space)))
    xs_ktx = np.zeros((len(search_space), len(search_space)))

    for i, w1_ in enumerate(search_space):
        print(f"\r{GREEN}Progress {(i+1) / len(search_space) * 100:.2f}%...{RESET}",
              end='', flush=True)
        for j, w2_ in enumerate(search_space):
            tmp_xs_nst = []
            tmp_xs_ktx = []
            tmp_zs = []
            tmp_j_nst = []
            tmp_j_ktx = []
            for k in range(1, 201):
                x_nst = min_x_NST(params, v, k, OD_out, OD_in, continuous=cont)
                x_ktx = min_x_KTX(params, v, k, OD_out, OD_in, continuous=cont)
                z = min_Z(params, v, k, OD_out, OD_in, continuous=cont)

                cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                              v=v, x=x_nst, y=k, z=z, stations=stations,
                                              OD_out=OD_out, OD_in=OD_in, mode='n_station',
                                              pareto=2, w1=w1_, w2=w2_)
                cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                              v=v, x=x_ktx, y=k, stations=stations,
                                              OD_out=OD_out, OD_in=OD_in, mode='n_station',
                                              pareto=2, w1=w1_, w2=w2_)
                tmp_xs_nst.append(x_nst)
                tmp_xs_ktx.append(x_ktx)
                tmp_zs.append(z)
                tmp_j_nst.append(cost_NST)
                tmp_j_ktx.append(cost_KTX)

            tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
            tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
            ys_nst[-(i+1),j] = tmp_idx_nst + 1
            xs_nst[-(i+1),j] = tmp_xs_nst[tmp_idx_nst]
            ys_ktx[-(i+1),j] = tmp_idx_ktx + 1
            xs_ktx[-(i+1),j] = tmp_xs_ktx[tmp_idx_ktx]
            zs[-(i+1),j] = np.sum(tmp_zs[tmp_idx_nst])
            J_nst[-(i+1),j] = tmp_j_nst[tmp_idx_nst][0]
            module_nst[-(i+1),j] = tmp_j_nst[tmp_idx_nst][1]
            electricity_nst[-(i+1),j] = tmp_j_nst[tmp_idx_nst][2]
            waiting_nst[-(i+1),j] = tmp_j_nst[tmp_idx_nst][3]
            travel_nst[-(i+1),j] = tmp_j_nst[tmp_idx_nst][4]

            J_ktx[-(i+1),j] = tmp_j_ktx[tmp_idx_ktx][0]
            module_ktx[-(i+1),j] = tmp_j_ktx[tmp_idx_ktx][1]
            electricity_ktx[-(i+1),j] = tmp_j_ktx[tmp_idx_ktx][2]
            waiting_ktx[-(i+1),j] = tmp_j_ktx[tmp_idx_ktx][3]
            travel_ktx[-(i+1),j] = tmp_j_ktx[tmp_idx_ktx][4]

    plt.imshow(J_nst, vmin=0, vmax=2.2e6)
    plt.title('w1 * agency_cost + w2 * user_cost (NST)')
    plt.ylabel('w1')
    plt.xlabel('w2')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 100.1, 20))
    ax.set_yticks(np.arange(0, 100.1, 20))
    ax.set_yticklabels(np.arange(10, -2, -2))
    ax.set_xticklabels(np.arange(0, 12, 2))
    plt.colorbar(label='Total cost ($/hr)')
    plt.tight_layout()
    # plt.savefig('figures/mu1_sensitivity.png')
    plt.show()

    plt.imshow(J_ktx, vmin=0, vmax=2.2e6)
    plt.title('w1 * agency_cost + w2 * user_cost (KTX)')
    plt.ylabel('w1')
    plt.xlabel('w2')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 100.1, 20))
    ax.set_yticks(np.arange(0, 100.1, 20))
    ax.set_yticklabels(np.arange(10, -2, -2))
    ax.set_xticklabels(np.arange(0, 12, 2))
    plt.colorbar(label='Total cost ($/hr)')
    plt.tight_layout()
    # plt.savefig('figures/mu1_sensitivity.png')
    plt.show()


    agency_cost_NST = module_nst + electricity_nst
    agency_cost_NST = agency_cost_NST.reshape([-1])
    user_cost_NST = waiting_nst + travel_nst
    user_cost_NST = user_cost_NST.reshape([-1])
    for i in range(30):
        plt.plot([-150000, 300000 + 150000 * i], [300000 + 150000 * i, -150000],
                 c='grey', alpha=0.4)
    plt.xlim([-10000, 275000])
    plt.ylim([-100000, 2100000])
    t_units_nst = xs_nst * ys_nst + zs
    t_units_nst = t_units_nst.reshape([-1])
    plt.scatter(agency_cost_NST, user_cost_NST,
                c=t_units_nst, cmap='viridis', vmin=195, vmax=320)
    plt.colorbar(label="Total number of units")
    # cbar.set_ticks([1, 20, 40, 60, 80, 100])
    # cbar.set_ticklabels(["1", "20", "40", "60", "80", ">100"])
    #
    plt.ylabel('user cost ($/hr)')
    plt.xlabel('agency cost ($/hr)')
    plt.title('pareto search - NST')
    plt.tight_layout()
    plt.show()

    agency_cost_KTX = module_ktx + electricity_ktx
    agency_cost_KTX = agency_cost_KTX.reshape([-1])
    user_cost_KTX = waiting_ktx + travel_ktx
    user_cost_KTX = user_cost_KTX.reshape([-1])
    for i in range(30):
        plt.plot([-150000, 300000 + 150000 * i], [300000 + 150000 * i, -150000],
                 c='grey', alpha=0.4)
    plt.xlim([-10000, 275000])
    plt.ylim([-100000, 2100000])
    t_units_ktx = xs_ktx * ys_ktx
    t_units_ktx = t_units_ktx.reshape([-1])
    plt.scatter(agency_cost_KTX, user_cost_KTX,
                c=t_units_ktx, cmap='viridis', vmin=195, vmax=320)
    plt.colorbar(label="Total number of units")
    # # cbar.set_ticks([1, 20, 40, 60, 80, 100])
    # # cbar.set_ticklabels(["1", "20", "40", "60", "80", ">100"])
    #
    plt.ylabel('user cost ($/hr)')
    plt.xlabel('agency cost ($/hr)')
    plt.title('pareto search - AST')
    plt.tight_layout()
    plt.show()


def plot_zs():
    plt.bar(np.arange(2, 11), [4, 2, 1, 1, 2, 1, 2, 1, 1])
    plt.xlabel('m')
    plt.ylabel(r'$\mathregular{z^{out}_i}$')
    plt.title('Number of decoupled units - outbound stations')
    ax = plt.gca()
    ax.set_xticks(np.arange(2, 11))
    ax.set_yticks(np.arange(0, 5))
    plt.savefig('figures/figure5a-outbound_Z')
    plt.show()

    plt.bar(np.arange(10, 1, -1), [1, 1, 2, 1, 2, 1, 1, 2, 4])
    plt.xlabel('m')
    plt.ylabel(r'$\mathregular{z^{in}_i}$')
    plt.title('Number of decoupled units - inbound stations')
    ax = plt.gca()
    ax.set_xticks(np.arange(2, 11))
    ax.set_yticks(np.arange(0, 5))
    plt.savefig('figures/figure5b-inbound_Z')
    plt.show()


def plot_demand_experiment(save=True):
    agency_costs_ktx = np.array([11795.9, 19739.83, 29090.28, 34588.77, 43235.96, 47961.54,
                                 55835.83, 61052.34, 68378.63, 75704.91, 80275.24, 87573])
    user_costs_ktx = np.array([48263.13, 93150.85, 135169.48, 180225.97,
                               222075.83,
                               267180.17,
                               309064.39,
                               353953.15,
                               395888.52,
                               437807.33,
                               482794.86,
                               524644.71])
    total_costs_ktx = agency_costs_ktx + user_costs_ktx
    ys_ktx = np.array([28, 38, 56, 56, 70, 67,
                       78, 75, 84, 93, 88, 96])
    xs_ktx = np.array([2, 3, 3, 4, 4, 5,
                       5, 6, 6, 6, 7, 7])
    t_units_ktx = np.array([56, 114, 168, 224, 280, 335,
                            390, 450, 504, 558, 616, 672])

    agency_costs_nst = np.array([17036.16, 26298.11, 37812.51, 42896.96, 47840.2, 57069.33,
                                 64760.26, 73989.37, 78531.63, 87042.4, 95553.16, 104063.93])
    user_costs_nst = np.array([36945.68, 70190.42, 100196.86, 136041.83, 170052.29, 199785.71,
                               230068.61, 259789.77, 293335.92, 323132.22, 352938.86, 382753.24])
    total_costs_nst = agency_costs_nst + user_costs_nst
    ys_nst = np.array([17, 23, 34, 29, 29, 35,
                       40, 46, 44, 49, 54, 59])
    xs_nst = np.array([3, 4, 4, 6, 7, 7,
                       7, 7, 8, 8, 8, 8])
    t_units_nst = np.array([71, 114, 158, 204, 235, 277,
                            312, 354, 388, 428, 468, 508])

    plt.plot(np.arange(0.25, 3.01, 0.25), total_costs_ktx,
             c='tab:blue', marker='o', label='Generalized (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), total_costs_nst,
             c='tab:orange', marker='o', label='Generalized (NST)')

    plt.plot(np.arange(0.25, 3.01, 0.25), agency_costs_ktx,
             '--', c='tab:blue', marker='*', label='Operator (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), agency_costs_nst,
             '--', c='tab:orange', marker='*', label='Operator (NST)')

    plt.plot(np.arange(0.25, 3.01, 0.25), user_costs_ktx,
             '-.', c='tab:blue', marker='^', label='User (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), user_costs_nst,
             '-.', c='tab:orange', marker='^', label='User (NST)')

    plt.ylim(0, 710000)
    ax = plt.gca()
    ax.set_yticks(np.arange(0, 700001, 100000))
    ax.set_yticklabels(np.arange(0, 701, 100))

    plt.ylabel('Cost (1,000 $/hr)')
    plt.xlabel('Demand scaling factor, q')
    for yy in np.arange(0, 700001, 100000):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0.5, 3.01, 0.5):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')

    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig('figures/figure9a-demand_scaling_costs.png')
    plt.show()

    savings_abs = total_costs_ktx - total_costs_nst
    savings_pct = savings_abs / total_costs_ktx * 100

    fig2, ax = plt.subplots(1, 1)
    ax.plot(np.arange(0.25, 3.01, 0.25), savings_abs, marker='o', label='Absolute savings ($/hr)')
    ax2 = ax.twinx()
    # ax2.plot(np.arange(0.25, 3.01, 0.25), savings_pct, marker='o', label='Relative savings (%)')
    ax2.bar(np.arange(0.25, 3.01, 0.25), savings_pct, width=0.2, color='red', alpha=0.2, label='Relative savings (%)')

    ax.set_xlabel('Demand scaling factor, q')
    ax.set_ylabel('Cost savings (1,000 $/hr)')
    ax2.set_ylabel('Cost savings (%)')
    # ax.set_title('NST total-cost savings vs demand scaling')
    ax.set_yticks(np.arange(25000, 135001, 25000))
    ax.set_yticklabels(np.arange(25, 136, 25))
    ax2.set_ylim(0, 25)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()

    if save:
        plt.savefig('figures/figure9b-nst_savings_vs_demand.png', dpi=200)
    plt.show()

    # fig2, ax = plt.subplots(1, 1)
    plt.plot(np.arange(0.25, 3.01, 0.25), t_units_ktx, c='tab:blue', marker='o', label='N. Units (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), t_units_nst, c='tab:orange', marker='o', label='N. Units (NST)')
    plt.xlabel('Demand scaling factor, q')
    plt.ylabel('Units')
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    for yy in np.arange(100, 701, 100):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0.5, 3.01, 0.5):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    if save:
        plt.savefig('figures/figure10a-total_units_vs_demand.png', dpi=200)
    plt.show()


    plt.plot(np.arange(0.25, 3.01, 0.25), ys_ktx, '-.', c='tab:blue', marker='*', label='N. trains (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), ys_nst, '-.', c='tab:orange', marker='*', label='N. trains (NST)')

    plt.plot(np.arange(0.25, 3.01, 0.25), xs_ktx, '--', c='tab:blue', marker='^', label='N. Units/train (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), xs_nst, '--', c='tab:orange', marker='^', label='N. Units/train (NST)')

    plt.xlabel('Demand scaling factor, q')
    plt.ylabel('Units')
    plt.legend()

    ax = plt.gca()
    for yy in np.arange(20, 100, 20):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0.5, 3.01, 0.5):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    if save:
        plt.savefig('figures/figure10b-decision_variables_vs_demand.png', dpi=200)
    plt.show()


def plot_demand_experiment_budget():
    agency_costs_ktx = np.array([11795.9, 20259.31, 29029.86, 36441.74, 43235.96, 51540.76,
                                 57983.35, 65857.64, 72300.24, 78961.04, 86287.32, 91882.85])
    user_costs_ktx = np.array([48263.13, 92908.51, 137011.59, 179573.77, 222075.83, 266063.43,
                               308467.53, 350413.76, 392650.13, 437011.09, 478983.2, 521311.37])
    total_costs_ktx = agency_costs_ktx + user_costs_ktx
    ys_ktx = np.array([28, 39, 47, 59, 70, 72,
                       81, 92, 101, 97, 106, 113])
    xs_ktx = np.array([2, 3, 4, 4, 4, 5,
                       5, 5, 5, 6, 6, 6])
    t_units_ktx = np.array([56, 117, 188, 236, 280, 360,
                            405, 460, 505, 582, 636, 678])

    agency_costs_nst = np.array([12147.66, 20658.43, 29059.08, 36801.74, 43514.55, 51763.74,
                                 58538.82, 66313.91, 72962.65, 79522.09, 86934.65, 94240.56])
    user_costs_nst = np.array([53969.95, 83822.18, 134777.41, 167644.35, 195641.38, 230794.2,
                               261723.12, 284756.35, 318001.76, 373890.17, 406402.43, 448668.19])
    total_costs_nst = agency_costs_nst + user_costs_nst
    ys_nst = np.array([5, 10, 8, 10, 13, 14,
                       16, 21, 22, 16, 17, 16])
    xs_nst = np.array([8, 8, 15, 16, 15, 17, 17, 15, 16, 24, 25, 29])
    t_units_nst = np.array([76, 116, 182, 226, 259, 310,
                            348, 379, 418, 486, 530, 582])

    # plt.plot(np.arange(0.25, 3.01, 0.25), total_costs_ktx,
    #          c='tab:blue', marker='o', label='Total (AST)')
    # plt.plot(np.arange(0.25, 3.01, 0.25), total_costs_nst,
    #          c='tab:orange', marker='o', label='Total (NST)')
    #
    # plt.plot(np.arange(0.25, 3.01, 0.25), agency_costs_ktx,
    #          '--', c='tab:blue', marker='*', label='Agency (AST)')
    # plt.plot(np.arange(0.25, 3.01, 0.25), agency_costs_nst,
    #          '--', c='tab:orange', marker='*', label='Agency (NST)')

    # plt.plot(np.arange(0.25, 3.01, 0.25), user_costs_ktx,
    #          c='tab:blue', marker='o', label='AST')
    # plt.plot(np.arange(0.25, 3.01, 0.25), user_costs_nst,
    #          c='tab:orange', marker='o', label='NST')

    plt.plot(np.arange(0.25, 3.01, 0.25), agency_costs_ktx,
             '--', c='tab:blue', marker='*', label='Operator (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), agency_costs_nst,
             '--', c='tab:orange', marker='*', label='Operator (NST)')

    plt.plot(np.arange(0.25, 3.01, 0.25), user_costs_ktx,
             '-.', c='tab:blue', marker='^', label='User (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), user_costs_nst,
             '-.', c='tab:orange', marker='^', label='User (NST)')


    ax = plt.gca()
    ax.set_yticks(np.arange(0, 700001, 100000))
    ax.set_yticklabels(np.arange(0, 701, 100))
    plt.ylim(0, 550000)

    plt.ylabel('cost (1,000 $/hr)')
    plt.xlabel('Demand scaling factor, q')
    plt.legend()
    for yy in np.arange(100e3, 501e3, 100e3):
        ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    for xx in np.arange(0.5, 3.1, 0.5):
        ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure16a-demand_scaling_budget_scenario.png')
    plt.show()

    savings_abs = user_costs_ktx - user_costs_nst
    savings_pct = savings_abs / user_costs_ktx * 100

    fig2, ax = plt.subplots(1, 1)
    ax.plot(np.arange(0.25, 3.01, 0.25), savings_abs, marker='o', label='Absolute savings ($/hr)')
    ax2 = ax.twinx()
    # ax2.plot(np.arange(0.25, 3.01, 0.25), savings_pct, marker='o', label='Relative savings (%)')
    ax2.bar(np.arange(0.25, 3.01, 0.25), -savings_pct, width=0.2, color='blue', alpha=0.2, label='Relative savings (%) (AST < NST)')
    ax2.bar(np.arange(0.25, 3.01, 0.25), savings_pct, width=0.2, color='red', alpha=0.2, label='Relative savings (%) (AST > NST)')


    ax.set_xlabel('Demand scaling factor, q')
    ax.set_ylabel('user cost savings (1,000 $/hr)')
    ax2.set_ylabel('user cost savings (%)')
    # ax.set_title('NST total-cost savings vs demand scaling')
    ax.set_yticks(np.arange(0, 90001, 25000))
    ax.set_yticklabels(np.arange(0, 91, 25))
    ax.set_ylim(-8000, 99000)
    ax2.set_ylim(0, 27)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    # for yy in np.arange(0, 76e3, 25e3):
    #     ax.axhline(yy, linewidth=0.8, alpha=0.3, color='grey')
    # for xx in np.arange(0.5, 3.1, 0.5):
    #     ax.axvline(xx, linewidth=0.8, alpha=0.3, color='grey')
    plt.tight_layout()
    plt.savefig('figures/figure16b-nst_savings_vs_demand_budget.png', dpi=200)
    plt.show()

    # fig2, ax = plt.subplots(1, 1)
    plt.plot(np.arange(0.25, 3.01, 0.25), t_units_ktx, c='tab:blue', marker='o', label='AST')
    plt.plot(np.arange(0.25, 3.01, 0.25), t_units_nst, c='tab:orange', marker='o', label='NST')
    plt.xlabel('Demand scaling factor, q')
    plt.ylabel('Total nits')
    plt.legend()
    plt.tight_layout()
    plt.savefig('total_units_vs_demand.png', dpi=200)
    plt.show()


    plt.plot(np.arange(0.25, 3.01, 0.25), ys_ktx, '-.', c='tab:blue', marker='*', label='N. trains (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), ys_nst, '-.', c='tab:orange', marker='*', label='N. trains (NST)')

    plt.plot(np.arange(0.25, 3.01, 0.25), xs_ktx, '--', c='tab:blue', marker='^', label='N. Units/train (AST)')
    plt.plot(np.arange(0.25, 3.01, 0.25), xs_nst, '--', c='tab:orange', marker='^', label='N. Units/train (NST)')

    plt.xlabel('Demand scaling factor, q')
    plt.ylabel('Units')
    plt.legend()
    # lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.savefig('decision_variables_vs_demand.png', dpi=200)
    plt.show()


def plot_v_assessment_budget():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, v_ in enumerate(np.arange(100, 410, 5)):
        tmp_ys_nst = []
        tmp_xs_nst = []
        tmp_ys_ktx = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        for j in range(1, 201):
            x_nst = min_x_NST(params, v_/3.6, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v_/3.6, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v_/3.6, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v_/3.6, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v_/3.6, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        # tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_j_nst = np.stack(tmp_j_nst)
        tmp_j_ktx = np.stack(tmp_j_ktx)
        tmp_agency_cost_ktx = np.sum(tmp_j_ktx[:, [1, 2, 5]], axis=1)
        tmp_user_cost_ktx = np.sum(tmp_j_ktx[:, 3:5], axis=1)
        tmp_agency_cost_nst = np.sum(tmp_j_nst[:, [1, 2, 5]], axis=1)
        tmp_user_cost_nst = np.sum(tmp_j_nst[:, 3:5], axis=1)

        tmp_agency_min_nst = np.min(tmp_agency_cost_nst)
        tmp_idx_nst = np.argmin(tmp_agency_cost_nst)

        # tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        tmp_candidate_idx_ktx = np.where(tmp_agency_cost_ktx <= tmp_agency_min_nst)[0]
        tmp_user_min_ktx = np.min(tmp_user_cost_ktx[tmp_candidate_idx_ktx])
        tmp_idx_ktx = np.where(tmp_user_cost_ktx == tmp_user_min_ktx)[0][0]
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    J_ktx = np.stack(J_ktx)
    J_nst = np.stack(J_nst)
    agency_cost_ktx = np.sum(J_ktx[:, [1, 2, 5]], axis=1)
    user_cost_ktx = np.sum(J_ktx[:, 3:5], axis=1)
    agency_cost_nst = np.sum(J_nst[:, [1, 2, 5]], axis=1)
    user_cost_nst = np.sum(J_nst[:, 3:5], axis=1)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.bar(np.arange(100, 410, 5), (user_cost_nst - user_cost_ktx) / user_cost_ktx * 100,
            width=4, alpha=0.2, color='blue', label='Savings (%) (AST < NST)')
    ax2.bar(np.arange(100, 410, 5), (user_cost_ktx - user_cost_nst) / user_cost_ktx * 100,
            width=4, alpha=0.2, color='red', label='Savings (%) (AST > NST)')

    ax2.set_ylabel('reduction rate (%)')
    ax2.set_ylim([0, 0.20])
    ax2.set_yticks(np.arange(0, 21, 5))
    # ax2.set_yticks(np.arange(0, 0.4 + 0.05, 0.1))

    ax1.plot(np.arange(100, 410, 5), user_cost_ktx, label='AST')
    ax1.plot(np.arange(100, 410, 5), user_cost_nst, label='NST', color='tab:orange')
    ax1.plot([50, 300], [user_cost_ktx[40], user_cost_ktx[40]], color='grey', linestyle='dashed')

    min_speed_idx = np.min(np.where(user_cost_nst < user_cost_ktx[40])[0])
    ax1.plot([100 + 5 * min_speed_idx, 100 + 5 * min_speed_idx],
             [user_cost_nst[min_speed_idx], 0], color='grey', linestyle='dashed')
    ax1.plot([300, 300], [user_cost_ktx[40], 0], color='grey', linestyle='dashed')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # fig.suptitle(r'Optimal cost by different $v_c$')
    ax1.set_ylabel('user cost (1,000 $/hr)')
    ax1.set_xlabel(r'cruising speed $v_c$ (km/h)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    # ax1.legend()

    ax1.set_yticks(np.arange(1.5e5, 4.5e5, 1e5))
    ax1.set_yticklabels(np.arange(150, 450, 100))
    ax1.set_xticks([100, 150, 200, int(100 + 5 * min_speed_idx), 250, 300, 350, 400])
    ax1.set_xticklabels([100, 150, 200, int(100 + 5 * min_speed_idx), 250, 300, 350, 400])
    ax1.set_ylim(100000, 410000)
    ax1.set_xlim(80, 420)

    plt.tight_layout()
    plt.savefig('figures/v_assessment.png')
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.bar(np.arange(100, 410, 5), (
            np.stack(J_ktx)[:, 0] - np.stack(J_nst)[:, 0]) / np.stack(J_ktx)[:, 0],
            width=4, alpha=0.2 , color='red')
    ax2.set_ylabel('reduction rate (%)')
    ax2.set_ylim([0, 0.4])
    ax2.set_yticks(np.arange(0, 0.4 + 0.05, 0.1))

    ax1.plot(np.arange(100, 410, 5), np.stack(J_ktx)[:, 0] - np.stack(J_nst)[:, 0])
    fig.suptitle(r'Cycle time difference')
    ax1.set_ylabel('cost (1,000 $/hr)')
    ax1.set_xlabel(r'$v_c$ (km/h)')
    ax1.set_yticks(np.arange(2.5e4, 6.5e4 + 1, 1e4))
    ax1.set_yticklabels(np.arange(25, 65 + 1, 10))

    plt.tight_layout()
    plt.savefig('figures/v_difference.png')
    plt.show()


def plot_mu_0_sensitivity_budget():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(1.0, 10, 19)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu01'] = m_ * params['mu00']
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        # tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        # tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        tmp_j_nst = np.stack(tmp_j_nst)
        tmp_j_ktx = np.stack(tmp_j_ktx)
        tmp_agency_cost_ktx = np.sum(tmp_j_ktx[:, [1, 2, 5]], axis=1)
        tmp_user_cost_ktx = np.sum(tmp_j_ktx[:, 3:5], axis=1)
        tmp_agency_cost_nst = np.sum(tmp_j_nst[:, [1, 2, 5]], axis=1)
        tmp_user_cost_nst = np.sum(tmp_j_nst[:, 3:5], axis=1)

        tmp_agency_min_nst = np.min(tmp_agency_cost_nst)
        tmp_idx_nst = np.argmin(tmp_agency_cost_nst)

        tmp_candidate_idx_ktx = np.where(tmp_agency_cost_ktx <= tmp_agency_min_nst)[0]
        tmp_user_min_ktx = np.min(tmp_user_cost_ktx[tmp_candidate_idx_ktx])
        tmp_idx_ktx = np.where(tmp_user_cost_ktx == tmp_user_min_ktx)[0][0]

        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    J_ktx = np.stack(J_ktx)
    J_nst = np.stack(J_nst)
    agency_cost_ktx = np.sum(J_ktx[:, [1, 2, 5]], axis=1)
    user_cost_ktx = np.sum(J_ktx[:, 3:5], axis=1)
    agency_cost_nst = np.sum(J_nst[:, [1, 2, 5]], axis=1)
    user_cost_nst = np.sum(J_nst[:, 3:5], axis=1)

    # plt.plot(np.linspace(1.0, 10, 19), user_cost_ktx, label='AST')
    # plt.plot(np.linspace(1.0, 10, 19), user_cost_nst, label='NST')
    plt.plot(np.linspace(1.0, 10, 19), J_ktx[:, 0], label='AST')
    plt.plot(np.linspace(1.0, 10, 19), J_nst[:, 0], label='NST')
    plt.plot([1.5, 1.5], [0, np.stack(J_ktx)[1, 0]], '--', c='grey')
    # plt.title(r'Optimal cost by different $m_0^1/m_0^0$')
    plt.ylabel('user cost (1,000 $/hr)')
    plt.xlabel(r'Relative train cost ratios ($\mu_0^1/\mu_0^0$)')
    ax = plt.gca()
    ax.set_yticks(np.arange(175000, 400001, 25000))
    ax.set_yticklabels(np.arange(175, 401, 25))
    plt.ylim([160000, 280000])
    ax.set_xticks(np.concatenate(([1, 1.5], np.arange(3, 10, 2), [10])))
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/mu0100_sensitivity.png')
    plt.show()


    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(params['mu00'] / 4, params['mu00'] * 3, 12)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu00'] = m_
        # params['mu41'] = 2.0 * params['mu40']
        params['mu01'] = m_ * 1.5
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        # tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        # tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        tmp_j_nst = np.stack(tmp_j_nst)
        tmp_j_ktx = np.stack(tmp_j_ktx)
        tmp_agency_cost_ktx = np.sum(tmp_j_ktx[:, [1, 2, 5]], axis=1)
        tmp_user_cost_ktx = np.sum(tmp_j_ktx[:, 3:5], axis=1)
        tmp_agency_cost_nst = np.sum(tmp_j_nst[:, [1, 2, 5]], axis=1)
        tmp_user_cost_nst = np.sum(tmp_j_nst[:, 3:5], axis=1)

        tmp_agency_min_nst = np.min(tmp_agency_cost_nst)
        tmp_idx_nst = np.argmin(tmp_agency_cost_nst)

        # tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        tmp_candidate_idx_ktx = np.where(tmp_agency_cost_ktx <= tmp_agency_min_nst)[0]
        tmp_user_min_ktx = np.min(tmp_user_cost_ktx[tmp_candidate_idx_ktx])
        tmp_idx_ktx = np.where(tmp_user_cost_ktx == tmp_user_min_ktx)[0][0]

        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    params['mu00'] = (labor_price + roll_train_pr)
    params['mu01'] = params['mu00'] * 1.5

    plt.plot(np.linspace(params['mu00'] / 4, params['mu00'] * 3, 12),
             np.stack(J_ktx)[:, 0], label='AST')
    plt.plot(np.linspace(params['mu00'] / 4, params['mu00'] * 3, 12),
             np.stack(J_nst)[:, 0], label='NST')
    plt.plot([params['mu00'], params['mu00']], [0, np.stack(J_ktx)[3, 0]], '--', c='grey')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # plt.title('Sensitivity of  cost by different m0')
    plt.ylabel('generalized cost (1,000 $/hr)')
    # plt.xlabel(r'$\mathregular{\mu_0^0}$'.join(' (default = 112.1USD/hr)'))
    plt.xlabel(r'train cost coefficient $\mathregular{\mu_0^0}$ (\$/hr)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 450000, 50000))
    ax.set_yticklabels(np.arange(50, 450, 50))
    ax.set_xticks(np.concatenate(([50, np.round(params['mu00'], 1)],
                                  np.arange(150, 351, 50))))

    plt.ylim([30000, 430000])
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/mu0_sensitivity.png')
    plt.show()

    # ols_x = np.arange(20, 400, 10)
    # ols_x = sm.add_constant(ols_x)
    # model_nst = sm.OLS(np.stack(J_nst)[:, 0], ols_x)
    # results_nst = model_nst.fit()
    # model_ktx = sm.OLS(np.stack(J_ktx)[:, 0], ols_x)
    # results_ktx = model_ktx.fit()
    # print(results_ktx.params)
    # print(results_nst.params)
    # plt.plot(np.arange(20, 400, 10), np.stack(J_nst)[:, 0], '.', label='NST', color='tab:blue')
    # plt.plot(np.arange(20, 400, 10), np.stack(J_ktx)[:, 0], '.', label='KTX', color='tab:orange')
    # plt.plot(np.arange(20, 400, 10), results_nst.predict(ols_x), color='tab:blue')
    # plt.plot(np.arange(20, 400, 10), results_ktx.predict(ols_x), color='tab:orange')
    # # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # plt.title('Optimal cost by different m0')
    # plt.ylabel('cost (1,000 $/hr)')
    # plt.xlabel('m0 ($/unit-hr)')
    # ax = plt.gca()
    # ax.set_yticks(np.arange(50000, 350000, 50000))
    # ax.set_yticklabels(np.arange(50, 350, 50))
    # plt.ylim([30000, 330000])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('figures/mu0_sensitivity.png')
    # plt.show()
    #
    # params['mu00'] = (labor_price + roll_train_pr)
    # params['mu01'] = params['mu00'] * 1.5


def plot_mu_1_sensitivity_budget():
    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(1.0, 10, 19)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu11'] = m_ * params['mu10']
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        # tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        # tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        tmp_j_nst = np.stack(tmp_j_nst)
        tmp_j_ktx = np.stack(tmp_j_ktx)
        tmp_agency_cost_ktx = np.sum(tmp_j_ktx[:, [1, 2, 5]], axis=1)
        tmp_user_cost_ktx = np.sum(tmp_j_ktx[:, 3:5], axis=1)
        tmp_agency_cost_nst = np.sum(tmp_j_nst[:, [1, 2, 5]], axis=1)
        tmp_user_cost_nst = np.sum(tmp_j_nst[:, 3:5], axis=1)

        tmp_agency_min_nst = np.min(tmp_agency_cost_nst)
        tmp_idx_nst = np.argmin(tmp_agency_cost_nst)

        # tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        tmp_candidate_idx_ktx = np.where(tmp_agency_cost_ktx <= tmp_agency_min_nst)[0]
        tmp_user_min_ktx = np.min(tmp_user_cost_ktx[tmp_candidate_idx_ktx])
        tmp_idx_ktx = np.where(tmp_user_cost_ktx == tmp_user_min_ktx)[0][0]

        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    params['mu10'] = (module_price + infra_price + roll_unit_pr)
    params['mu11'] = params['mu10'] * 2

    J_ktx = np.stack(J_ktx)
    J_nst = np.stack(J_nst)
    agency_cost_ktx = np.sum(J_ktx[:, [1, 2, 5]], axis=1)
    user_cost_ktx = np.sum(J_ktx[:, 3:5], axis=1)
    agency_cost_nst = np.sum(J_nst[:, [1, 2, 5]], axis=1)
    user_cost_nst = np.sum(J_nst[:, 3:5], axis=1)

    idx = np.min(np.where(np.stack(J_ktx)[:, 0] < np.stack(J_nst)[:, 0])[0])
    val = np.linspace(1, 10, 19)[idx]
    plt.plot(np.linspace(1.0, 10, 19), J_ktx[:, 0], label='AST')
    plt.plot(np.linspace(1.0, 10, 19), J_nst[:, 0], label='NST')
    # plt.plot([val, val], [0, np.stack(J_ktx)[idx, 0]], c='black')
    plt.plot([2, 2], [0, J_ktx[2, 0]], '--', c='grey')
    plt.ylabel('generalized cost (1,000 $/hr)')
    plt.xlabel(r'Relative unit cost ratios ($\mu_1^1/\mu_1^0$)')
    ax = plt.gca()
    ax.set_yticks(np.arange(175000, 400001, 25000))
    ax.set_yticklabels(np.arange(175, 401, 25))
    plt.ylim([160000, 280000])
    ax.set_xticks(np.concatenate(([1, 2], np.arange(3, 10, 2), [10])))

    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/mu1110_sensitivity.png')
    plt.show()


    plt.plot(np.linspace(1.0, 10, 19), agency_cost_ktx, label='AST')
    plt.plot(np.linspace(1.0, 10, 19), agency_cost_nst, label='NST')
    # plt.plot([val, val], [0, np.stack(J_ktx)[idx, 0]], c='black')
    plt.plot([2, 2], [0, J_ktx[2, 0]], '--', c='grey')
    plt.ylabel('generalized cost (1,000 $/hr)')
    plt.xlabel(r'Relative unit cost ratios ($\mu_1^1/\mu_1^0$)')
    ax = plt.gca()
    # ax.set_yticks(np.arange(175000, 400001, 25000))
    # ax.set_yticklabels(np.arange(175, 401, 25))
    # plt.ylim([160000, 280000])
    # ax.set_xticks(np.concatenate(([1, 2], np.arange(3, 10, 2), [10])))

    plt.legend()
    plt.tight_layout()
    # plt.savefig('figures/mu1110_sensitivity.png')
    plt.show()

    J_nst = []
    J_ktx = []
    ys_nst = []
    xs_nst = []
    ys_ktx = []
    xs_ktx = []
    zs = []
    for i, m_ in enumerate(np.linspace(params['mu10'] / 4, params['mu10'] * 3, 12)):
        tmp_xs_nst = []
        tmp_xs_ktx = []
        tmp_zs = []
        tmp_j_nst = []
        tmp_j_ktx = []
        params['mu10'] = m_
        params['mu11'] = 2.0 * params['mu10']
        for j in range(1, 201):
            x_nst = min_x_NST(params, v, j, OD_out, OD_in, continuous=cont)
            x_ktx = min_x_KTX(params, v, j, OD_out, OD_in, continuous=cont)
            z = min_Z(params, v, j, OD_out, OD_in, continuous=cont)

            cost_NST = calculate_cost_NST(coeffs=coefficients, params=params,
                                          v=v, x=x_nst, y=j, z=z, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            cost_KTX = calculate_cost_KTX(coeffs=coefficients, params=params,
                                          v=v, x=x_ktx, y=j, stations=stations,
                                          OD_out=OD_out, OD_in=OD_in, mode='n_station')
            tmp_xs_nst.append(x_nst)
            tmp_xs_ktx.append(x_ktx)
            tmp_zs.append(z)
            tmp_j_nst.append(cost_NST)
            tmp_j_ktx.append(cost_KTX)

        tmp_idx_nst = np.argmin(np.stack(tmp_j_nst)[:, 0])
        tmp_idx_ktx = np.argmin(np.stack(tmp_j_ktx)[:, 0])
        ys_nst.append(tmp_idx_nst + 1)
        xs_nst.append(tmp_xs_nst[tmp_idx_nst])
        ys_ktx.append(tmp_idx_ktx + 1)
        xs_ktx.append(tmp_xs_ktx[tmp_idx_ktx])
        zs.append(tmp_zs[tmp_idx_nst])
        J_nst.append(tmp_j_nst[tmp_idx_nst])
        J_ktx.append(tmp_j_ktx[tmp_idx_ktx])

    params['mu10'] = (module_price + infra_price + roll_unit_pr)
    params['mu11'] = params['mu10'] * 2

    plt.plot(np.linspace(params['mu10'] / 4, params['mu10'] * 3, 12),
             np.stack(J_ktx)[:, 0], label='AST')
    plt.plot(np.linspace(params['mu10'] / 4, params['mu10'] * 3, 12),
             np.stack(J_nst)[:, 0], label='NST')
    plt.plot([params['mu10'], params['mu10']], [0, np.stack(J_ktx)[4, 0]], '--', c='grey')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    # plt.title('Optimal cost by different m10')
    plt.ylabel('generalized cost (1,000 $/hr)')
    plt.xlabel(r'unit cost coefficient $\mathregular{\mu_1^0}$ (\$/unit-hr)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 450000, 50000))
    ax.set_yticklabels(np.arange(50, 450, 50))
    plt.ylim([30000, 430000])
    ax.set_xticks(np.concatenate(([20, 40, np.round(params['mu10'])], np.arange(60, 141, 20))))

    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/mu1_sensitivity.png')
    plt.show()

    ols_x = np.arange(2, 42, 2)
    ols_x = sm.add_constant(ols_x)
    model_nst = sm.OLS(np.stack(J_nst)[:, 0], ols_x)
    results_nst = model_nst.fit()
    model_ktx = sm.OLS(np.stack(J_ktx)[:, 0], ols_x)
    results_ktx = model_ktx.fit()
    print(results_ktx.params)
    print(results_nst.params)
    plt.plot(np.arange(2, 42, 2), np.stack(J_nst)[:, 0], '.', label='NST', color='tab:blue')
    plt.plot(np.arange(2, 42, 2), np.stack(J_ktx)[:, 0], '.', label='KTX', color='tab:orange')
    plt.plot(np.arange(2, 42, 2), results_nst.predict(ols_x), color='tab:blue')
    plt.plot(np.arange(2, 42, 2), results_ktx.predict(ols_x), color='tab:orange')
    # plt.plot([100, 400], [J_ktx[20][0], J_ktx[20][0]], '--', c='grey')
    plt.title('Optimal cost by different m10')
    plt.ylabel('cost (1,000 $/hr)')
    plt.xlabel('m10 ($/unit-hr)')
    ax = plt.gca()
    ax.set_yticks(np.arange(50000, 350000, 50000))
    ax.set_yticklabels(np.arange(50, 350, 50))
    plt.ylim([30000, 330000])
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/mu1_sensitivity.png')
    plt.show()

    params['mu10'] = (module_price + infra_price + roll_unit_pr)
    params['mu11'] = params['mu10'] * 2


def plot_budget_default():
    operator_AST = 36441.74
    user_AST = 179573.77
    total_AST = 216015.51

    operator_NST = 36801.74
    user_NST = 167644.35
    total_NST = 204446.09

    fig, ax = plt.subplots()
    ax.bar(np.arange(2) - 0.1, [operator_AST, user_AST], width=0.2, label='AST')
    ax.bar(np.arange(2) + 0.1, [operator_NST, user_NST], width=0.2, label='NST')

    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['operator cost', 'user cost'], fontsize=14)
    ax.set_ylabel('Cost ($1,000/hr)')
    ax.set_yticks(np.arange(0, 1.8e5, 0.25e5))
    ax.set_yticklabels(np.arange(0, 180, 25))
    ax.legend()
    plt.xlim(-0.5, 1.5)
    # ax.grid()
    plt.tight_layout()
    plt.savefig('figures/figure15-budget_scenario_results.png')
    plt.show()


def plot_demand():

    OD = OD_out + OD_in
    OD[OD == 0] = np.nan
    plt.imshow(OD)
    stations = ['Haengsin', 'Seoul', 'Gwangmyeong', 'Cheonan-Asan', 'Osong',
                'Daejeon', 'Gimchum (Gumi)', 'Dongdaegu', 'Gyeongju', 'Ulsan', 'Busan']
    stations_abbrev = ['HA', 'SE', 'GW', 'CH', 'OS', 'DA', 'GI', 'DO', 'GY', 'UL', 'BU']
    plt.xticks(np.arange(len(stations_abbrev)), stations_abbrev)
    plt.yticks(np.arange(len(stations)), stations)
    cbar = plt.colorbar()
    cbar.set_label('Hourly demand (pax/hr)')
    plt.tight_layout()
    plt.savefig('figures/ODdemand.png')
    plt.show()


# plot_xyz()
# plot_zs()
# plot_total_cost()
# # plot_indiv_cost_10_100()
# plot_indiv_cost()
# plot_v_assessment()
# plot_energy_cost()
# plot_mu_0_sensitivity()
# plot_mu_1_sensitivity()
# plot_mu_2_sensitivity()
# plot_mu_3_sensitivity()
# pareto_search()
# pareto_2d()


