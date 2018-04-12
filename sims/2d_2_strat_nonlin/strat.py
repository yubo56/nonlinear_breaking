'''
2d_1_strat + Navier-Stokes dissipation
'''
from multiprocessing import Pool
import numpy as np
import strat_helper

N_PARALLEL = 8
H = 1
num_timesteps = 1e4

XMAX = H
ZMAX = 2 * H
KX = -2 * np.pi / H
KZ = -(np.pi / 2) * np.pi / H
G = (KX**2 + KZ**2 + 1 / (4 * H**2)) / KX**2 * (2 * np.pi)**2 * H # omega = 2pi
OMEGA = strat_helper.get_omega(G, H, KX, KZ)
VPH_X, VPH_Z = strat_helper.get_vph(G, H, KX, KZ)
T_F = abs(ZMAX / VPH_Z) * 4
DT = T_F / num_timesteps
RHO0 = 1

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 64,
              'N_Z': 256,
              'T_F': T_F,
              'DT': DT,
              'KX': KX,
              'KZ': KZ,
              'H': H,
              'RHO0': RHO0,
              'G': G,
              'A': 0.005,
              'NUM_SNAPSHOTS': 200}

def build_interp_params(interp_x, interp_z, dt=DT, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = dt
    params['NU'] = 0.3 * (ZMAX / params['N_Z'])**2 / np.pi**2 # smallest wavenumber
    return params

def get_sponge(domain):
    sponge_strength = 3
    zmax = PARAMS_RAW['ZMAX']
    damp_start = zmax * 0.7 # start damping zone
    z = domain.grid(1)

    # sponge field
    sponge = domain.new_field()
    sponge.meta['x']['constant'] = True
    sponge['g'] = sponge_strength * np.maximum(
        1 - (z - zmax)**2 / (damp_start - zmax)**2,
        np.zeros(np.shape(z)))
    return sponge

def sponge_lin(problem, domain):
    '''
    puts a -gamma(z) * q damping on all dynamical variables, where gamma(z)
    is the sigmoid: damping * exp(steep * (z - z_sigmoid)) / (1 + exp(...))

    w/o nonlin terms
    '''
    problem.parameters['sponge'] = get_sponge(domain)
    problem.add_equation('dx(ux) + uz_z = 0')
    problem.add_equation('dt(rho) - rho0 * uz / H = 0')
    problem.add_equation(
        'dt(ux) + dx(P) / rho0 - NU * (dz(ux_z) + dx(dx(ux))) + sponge * ux = 0')
    problem.add_equation(
        'dt(uz) + dz(P) / rho0 + rho * g / rho0 - NU * (dz(uz_z) + dx(dx(uz))) + sponge * uz = 0')
    problem.add_equation('dz(ux) - ux_z = 0')
    problem.add_equation('dz(uz) - uz_z = 0')

    problem.add_bc('left(P) = 0', condition='nx == 0')
    problem.add_bc('left(uz) = A * cos(KX * x - omega * t)')
    problem.add_bc('left(ux) = -KZ / KX * A * cos(KX * x - omega * t)')
    problem.add_bc('right(uz) = 0', condition='nx != 0')
    problem.add_bc('right(ux) = 0')

def sponge_nonlin(problem, domain):
    '''
    sponge zone velocities w nonlin terms
    '''
    problem.parameters['sponge'] = get_sponge(domain)
    problem.add_equation('dx(ux) + uz_z = 0')
    problem.add_equation('dt(rho) = -ux * dx(rho) - uz * dz(rho)')
    problem.add_equation(
        'dt(ux) - NU * (dz(ux_z) + dx(dx(ux))) + sponge * ux + dx(P) / rho0' +
        '= - dx(P) / rho + dx(P) / rho0 - ux * dx(ux) - uz * ux_z')
    problem.add_equation(
        'dt(uz) - NU * (dz(uz_z) + dx(dx(uz))) + sponge * uz + dz(P) / rho0' +
        '= -g - dz(P) / rho + dz(P) / rho0- ux * dx(uz) - uz * uz_z')
    problem.add_equation('dz(ux) - ux_z = 0')
    problem.add_equation('dz(uz) - uz_z = 0')

    problem.add_bc('left(P) = 0', condition='nx == 0')
    problem.add_bc('left(uz) = A * cos(KX * x - omega * t)')
    problem.add_bc('left(ux) = -KZ / KX * A * cos(KX * x - omega * t)')
    problem.add_bc('right(uz) = 0', condition='nx != 0')
    problem.add_bc('right(ux) = 0')

def zero_ic(solver, domain):
    ux = solver.state['ux']
    uz = solver.state['uz']
    P = solver.state['P']
    rho = solver.state['rho']
    gshape = domain.dist.grid_layout.global_shape(scales=1)

    P['g'] = np.zeros(gshape)
    ux['g'] = np.zeros(gshape)
    uz['g'] = np.zeros(gshape)
    rho['g'] = np.zeros(gshape)

def bg_ic(solver, domain):
    ux = solver.state['ux']
    uz = solver.state['uz']
    P = solver.state['P']
    rho = solver.state['rho']
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    z = domain.grid(1)

    ux['g'] = np.zeros(gshape)
    uz['g'] = np.zeros(gshape)
    rho['g'] = RHO0 * np.exp(-z / H)
    P['g'] = RHO0 * (np.exp(-z / H) - 1) * G * H

def run(bc, ic, name, params_dict):
    assert 'INTERP_X' in params_dict and 'INTERP_Z' in params_dict,\
        'params need INTERP'
    strat_helper.run_strat_sim(bc, ic, name=name, **params_dict)
    return '%s completed' % name

if __name__ == '__main__':
    tasks = [
        (sponge_lin, zero_ic, 'sponge_lin', build_interp_params(8, 8)),
        (sponge_nonlin, bg_ic, 'sponge_nonlin', build_interp_params(8, 8)),
        (sponge_lin, zero_ic, 'sponge_highA_lin',
         build_interp_params(8, 8, overrides={'A': 0.04})),
        (sponge_nonlin, bg_ic, 'sponge_highA_nonlin',
         build_interp_params(8, 8, overrides={'A': 0.04})),
        # (rad_bc, zero_ic, 'rad', build_interp_params(8, 4)),
    ]

    with Pool(processes=N_PARALLEL) as p:
        res = []
        for task in tasks:
            res.append(p.apply_async(run, task))

        for r in res:
            print(r.get())

    for bc, _, name, params_dict in tasks:
        strat_helper.plot(bc, name=name, **params_dict)
