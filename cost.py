import numpy as np
# from osgeo.ogr import wkbTIN

from energy_consumption import *
from cycle_times import *

def calculate_cost_NST(coeffs, params, v, x, y, z, stations, OD_out, OD_in,
                       mode='n_station', pareto=0, w1=1.0, w2=1.0):
    train_cost = params['mu01'] * y
    module_cost = params['mu11'] * (x * y + np.sum(z))

    energy_cost = (params['mu2'] / (headway_NST(params, v, y) / 3600) *
                   energy_NST(coeffs, params, v, x, Z=z, mode=mode))
    wt_time, ivtt_time = passenger_travel_times_NST(params, v, y, stations, OD_out, OD_in)
    waiting_cost = 2 * params['mu3'] * wt_time
    invehicle_cost = params['mu3'] * ivtt_time

    if pareto == 0:
        total_cost = module_cost + energy_cost + waiting_cost + invehicle_cost + train_cost
        return total_cost, module_cost, energy_cost, waiting_cost, invehicle_cost, train_cost
    elif pareto == 1:
        total_cost = w1 * (module_cost + energy_cost + train_cost) + \
                     (1 - w1) * (waiting_cost + invehicle_cost)
        return total_cost, module_cost, energy_cost, waiting_cost, invehicle_cost, train_cost

    elif pareto == 2:
        total_cost = w1 * (module_cost + energy_cost + train_cost) + \
                     w2 * (waiting_cost + invehicle_cost)
        return total_cost, module_cost, energy_cost, waiting_cost, invehicle_cost, train_cost
    else:
        raise NotImplementedError('pareto mode not implemented')

def calculate_cost_KTX(coeffs, params, v, x, y, stations, OD_out, OD_in,
                       mode='n_station', pareto=0, w1=1.0, w2=1.0):
    train_cost = params['mu00'] * y
    module_cost = params['mu10'] * (x * y)
    energy_cost = (params['mu2'] / (headway_KTX(params, v, y) / 3600) *
                   energy_KTX(coeffs, params, v, x, mode=mode))
    wt_time, ivtt_time = passenger_travel_times_KTX(params, v, y, stations, OD_out, OD_in)
    waiting_cost = 2 * params['mu3'] * wt_time
    invehicle_cost = params['mu3'] * ivtt_time

    if pareto == 0:
        total_cost = module_cost + energy_cost + waiting_cost + invehicle_cost + train_cost
        return total_cost, module_cost, energy_cost, waiting_cost, invehicle_cost, train_cost
    elif pareto == 1:
        total_cost = w1 * (module_cost + energy_cost + train_cost) + \
                     (1 - w1) * (waiting_cost + invehicle_cost)
        return total_cost, module_cost, energy_cost, waiting_cost, invehicle_cost, train_cost
    elif pareto == 2:
        total_cost = w1 * (module_cost + energy_cost + train_cost) + \
                     w2 * (waiting_cost + invehicle_cost)
        return total_cost, module_cost, energy_cost, waiting_cost, invehicle_cost, train_cost
    else:
        raise NotImplementedError('pareto mode not implemented')

def headway_NST(params, v, y):
    h_min = params['alpha'] * v / params['a']
    h_p = cycle_NST(params, v) / y
    h = np.max([h_p, h_min])
    return h

def headway_KTX(params, v, y):
    h_min = params['alpha'] * v / params['a']
    h_p = cycle_conventional(params, v) / y
    h = np.max([h_p, h_min])
    return h

def passenger_travel_times_NST(params, v, y, stations, OD_out, OD_in):
    wt_cost = 0
    ivtt_cost = 0

    wt = headway_NST(params, v, y) / 2
    for i in range(len(stations)-1):
        for j in range(1, len(stations)):
            if i < j:
                s = stations[j] - stations[i]
                ivtt_ij = s / v + v / params['a']
                lambda_ij = OD_out[i, j]
                ivtt_cost_ij = lambda_ij * ivtt_ij
                wt_cost_ij = lambda_ij * wt
                wt_cost += wt_cost_ij
                ivtt_cost += ivtt_cost_ij

    for i in range(1, len(stations)):
        for j in range(len(stations)-1):
            if i > j:
                s = stations[i] - stations[j]
                ivtt_ij = s / v + v / params['a']
                lambda_ij = OD_in[i, j]
                # if i == 3:
                #     print(y)
                #     print(lambda_ij, wt, ivtt_ij)
                ivtt_cost_ij = lambda_ij * ivtt_ij
                wt_cost_ij = lambda_ij * wt
                wt_cost += wt_cost_ij
                ivtt_cost += ivtt_cost_ij

    # print('s', pt_cost / 3600)
    # import sys
    # sys.exit()
    return wt_cost / 3600, ivtt_cost / 3600

def passenger_travel_times_KTX(params, v, y, stations, OD_out, OD_in):
    wt_cost = 0
    ivtt_cost = 0

    wt = headway_KTX(params, v, y) / 2
    for i in range(len(stations)-1):
        for j in range(1, len(stations)):
            if i < j:
                s = stations[j] - stations[i]
                ivtt_ij = s / v + (j - i) * (params['ts'] + v / params['a']) - params['ts']
                lambda_ij = OD_out[i, j]
                ivtt_cost_ij = lambda_ij * ivtt_ij
                wt_cost_ij = lambda_ij * wt
                wt_cost += wt_cost_ij
                ivtt_cost += ivtt_cost_ij

    for i in range(1, len(stations)):
        for j in range(len(stations)-1):
            if i > j:
                s = stations[i] - stations[j]
                # ivtt_ij = s / v + v / params['a']
                ivtt_ij = s / v + (i - j) * (params['ts'] + v / params['a']) - params['ts']
                lambda_ij = OD_in[i, j]
                ivtt_cost_ij = lambda_ij * ivtt_ij
                wt_cost_ij = lambda_ij * wt
                wt_cost += wt_cost_ij
                ivtt_cost += ivtt_cost_ij

    return wt_cost / 3600, ivtt_cost / 3600

def min_Z(params, v, y, OD_out, OD_in, continuous=False):
    zs_out = []
    h = headway_NST(params, v, y) / 3600

    for i in range(1, len(OD_out)-1):
        if continuous:
            z_out_i_d = h * np.sum(OD_out[:i, i]) / params['c']
            z_out_i_o = h * np.sum(OD_out[i, i + 1:]) / params['c']
        else:
            z_out_i_d = np.ceil(h * np.sum(OD_out[:i, i]) / params['c'])
            z_out_i_o = np.ceil(h * np.sum(OD_out[i, i+1:]) / params['c'])
        z_out_i = np.max([z_out_i_d, z_out_i_o])

        zs_out.append(z_out_i)

    zs_in = []
    for i in range(len(OD_in)-2, 0, -1):
        if continuous:
            z_in_i_d = h * np.sum(OD_in[i+1:, i]) / params['c']
            z_in_i_o = h * np.sum(OD_in[i, :i]) / params['c']
        else:
            z_in_i_d = np.ceil(h * np.sum(OD_in[i + 1:, i]) / params['c'])
            z_in_i_o = np.ceil(h * np.sum(OD_in[i, :i]) / params['c'])
        z_in_i = np.max([z_in_i_d, z_in_i_o])
        zs_in.append(z_in_i)
    zs_out.extend(zs_in)
    return zs_out


def min_x_NST(params, v, y, OD_out, OD_in, continuous=False):
    h = headway_NST(params, v, y) / 3600.0

    xs_out = []

    if continuous:
        x_out_0 = np.ceil(h * np.sum(OD_out[0, 1:]) / params['c'])
    else:
        x_out_0 = h * np.sum(OD_out[0, 1:]) / params['c']
    xs_out.append(x_out_0)
    for i in range(1, len(OD_out)-1):
        x_out_i_p = np.sum(OD_out[:i, i+1:])
        z_out_i_d = np.sum(OD_out[:i, i])
        z_out_i_o = np.sum(OD_out[i, i + 1:])
        if continuous:
            z_out_i = h * np.max([z_out_i_d, z_out_i_o]) / params['c']
            xs_out_i = h * x_out_i_p / params['c']
        else:
            z_out_i = np.ceil(h * np.max([z_out_i_d, z_out_i_o]) / params['c'])
            xs_out_i = np.ceil(h * x_out_i_p / params['c'])
        xs_out_i = xs_out_i + z_out_i
        xs_out.append(xs_out_i)
    if continuous:
        x_out_M = h * np.sum(OD_in[:-1, -1]) / params['c']
    else:
        x_out_M = np.ceil(h * np.sum(OD_in[:-1, -1]) / params['c'])
    xs_out.append(x_out_M)

    xs_in = []
    if continuous:
        x_in_0 = h * np.sum(OD_in[-1, :-1]) / params['c']
    else:
        x_in_0 = np.ceil(h * np.sum(OD_in[-1, :-1]) / params['c'])

    xs_in.append(x_in_0)
    for i in range(len(OD_in)-2, 0, -1):
        x_in_i_p = np.sum(OD_in[i+1:, :i])
        z_in_i_d = np.sum(OD_in[i+1:, i])
        z_in_i_o = np.sum(OD_in[i, :i])
        if continuous:
            z_in_i = h * np.max([z_in_i_d, z_in_i_o]) / params['c']
            xs_in_i = h * x_in_i_p / params['c']
        else:
            z_in_i = np.ceil(h * np.max([z_in_i_d, z_in_i_o]) / params['c'])
            xs_in_i = np.ceil(h * x_in_i_p / params['c'])
        xs_in_i = xs_in_i + z_in_i

        xs_in.append(xs_in_i)
    if continuous:
        xs_in_M = h * np.sum(OD_in[1:, 0]) / params['c']
    else:
        xs_in_M = np.ceil(h * np.sum(OD_in[1:, 0]) / params['c'])
    xs_in.append(xs_in_M)
    xs_out.extend(xs_in)

    return np.max(xs_out)


def min_x_KTX(params, v, y, OD_out, OD_in, continuous=False):
    h = headway_KTX(params, v, y) / 3600.0

    xs_out = []

    x_out_0 = h * np.sum(OD_out[0, 1:]) / params['c']
    if not continuous:
        x_out_0 = np.ceil(h * np.sum(OD_out[0, 1:]) / params['c'])

    xs_out.append(x_out_0)
    for i in range(1, len(OD_out)-1):
        x_out_i_p = np.sum(OD_out[:i+1, i+1:])
        xs_out_i = h * x_out_i_p / params['c']
        if not continuous:
            xs_out_i = np.ceil(h * x_out_i_p / params['c'])
        xs_out.append(xs_out_i)

    x_out_M = h * np.sum(OD_in[:-1, -1]) / params['c']
    if not continuous:
        x_out_M = np.ceil(h * np.sum(OD_in[:-1, -1]) / params['c'])
    xs_out.append(x_out_M)

    xs_in = []

    x_in_0 = h * np.sum(OD_in[-1, :-1]) / params['c']
    if not continuous:
        x_in_0 = np.ceil(h * np.sum(OD_in[-1, :-1]) / params['c'])

    xs_in.append(x_in_0)
    for i in range(len(OD_in)-2, 0, -1):
        x_in_i_p = np.sum(OD_in[i+1:, :i])

        xs_in_i = h * x_in_i_p / params['c']
        if not continuous:
            xs_in_i = np.ceil(h * x_in_i_p / params['c'])
        xs_in.append(xs_in_i)

    xs_in_M = h * np.sum(OD_in[1:, 0]) / params['c']
    if not continuous:
        xs_in_M = np.ceil(h * np.sum(OD_in[1:, 0]) / params['c'])
    xs_in.append(xs_in_M)
    xs_out.extend(xs_in)

    return np.max(xs_out)