'''
Incompressible fluid equations w/ vertical stratification
- div u1 = 0
- dt (rho1) - rho0 * u1z / H = 0
- dt(u1) = -grad(P1) / rho0 - rho1 * g / rho0

Periodic BC in x, Dirichlet/Neumann 0 at z=L, Driving term at z=0
creates h5 snapshot, then plots. if snapshot exists, skips computation
'''
from multiprocessing import Pool
import numpy as np
import strat_helper
import matplotlib.pyplot as plt

N_PARALLEL = 8
H = 1
num_timesteps = 1e4

XMAX = H
ZMAX = 5 * H
KX = -2 * np.pi / H
KZ = (np.pi / 2) * np.pi / H
G = 10
OMEGA = strat_helper.get_omega(G, H, KX, KZ)
VPH_X, VPH_Z = strat_helper.get_vph(G, H, KX, KZ)
T_F = (ZMAX / VPH_Z) * 4
DT = T_F / num_timesteps

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 64,
              'N_Z': 512,
              'T_F': T_F,
              'DT': DT,
              'KX': KX,
              'KZ': KZ,
              'H': H,
              'RHO0': 1,
              'G': G,
              'A': 0.05,
              'NUM_SNAPSHOTS': 200}

def build_interp_params(interp_x, interp_z, dt=None):
    params = dict(PARAMS_RAW)
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = dt or DT
    params['USE_CFL'] = dt == None
    return params

def dirichlet_bc(problem, *_):
    strat_helper.default_problem(problem)
    problem.add_bc('right(uz) = 0', condition='nx != 0')

def neumann_bc(problem, *_):
    strat_helper.default_problem(problem)
    problem.add_bc('right(dz(uz)) = 0', condition='nx != 0')

def rad_bc(problem, *_):
    strat_helper.default_problem(problem)
    problem.add_bc('right(dt(uz) + omega / KZ * dz(uz)) = 0',
                   condition='nx != 0')

def sponge(problem, domain):
    '''
    puts a -gamma(z) * q damping on all dynamical variables, where gamma(z)
    is the sigmoid: damping * exp(steep * (z - z_sigmoid)) / (1 + exp(...))
    '''
    zmax = PARAMS_RAW['ZMAX']
    damp_start = zmax * 0.7 # start damping zone
    z = domain.grid(1)

    # sponge field
    sponge = domain.new_field()
    sponge.meta['x']['constant'] = True
    sponge['g'] = np.maximum(
        1 - (z - zmax)**2 / (damp_start - zmax)**2,
        np.zeros(np.shape(z)))

    problem.parameters['sponge'] = sponge
    problem.add_equation("dx(ux) + dz(uz) = 0")
    problem.add_equation("dt(rho) - rho0 * uz / H + sponge * rho= 0")
    problem.add_equation("dt(ux) + dx(P) / rho0 + sponge * ux= 0")
    problem.add_equation(
        "dt(uz) + dz(P) / rho0 + rho * g / rho0 + sponge * uz= 0")

    problem.add_bc("left(P) = 0", condition="nx == 0")
    problem.add_bc("left(uz) = cos(KX * x - omega * t)")
    problem.add_bc('right(uz) = 0', condition='nx != 0')

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

def run(name, args):
    bc, ic, params_dict = args
    assert 'INTERP_X' in params_dict and 'INTERP_Z' in params_dict,\
        'params need INTERP'
    strat_helper.run_strat_sim(bc, ic, name=name, **params_dict)
    return '%s completed' % name

def rms_diff(arr1, arr2):
    num_times = len(arr1)
    return np.sqrt(np.sum((arr1[num_times // 2: ] - arr2[num_times // 2: ])**2))\
        / np.sqrt(np.sum(arr1[num_times //2 : ]**2))

if __name__ == '__main__':
    tasks = {
        'd0_dt0': (dirichlet_bc, zero_ic, build_interp_params(16, 8, dt=DT * 2)),
        'd0_dt2': (dirichlet_bc, zero_ic, build_interp_params(16, 8, dt=DT / 2)),
        'd0_dt3': (dirichlet_bc, zero_ic, build_interp_params(16, 8, dt=DT / 4)),
        'd0_dt4': (dirichlet_bc, zero_ic, build_interp_params(16, 8, dt=DT / 8)),
        'd0_dt5': (dirichlet_bc, zero_ic, build_interp_params(16, 8, dt=DT / 16)),
        'd0_16_1': (dirichlet_bc, zero_ic, build_interp_params(16, 1, dt=DT)),
        'd0_16_2': (dirichlet_bc, zero_ic, build_interp_params(16, 2, dt=DT)),
        'd0_16_4': (dirichlet_bc, zero_ic, build_interp_params(16, 4, dt=DT)),
        'd0_16_8': (dirichlet_bc, zero_ic, build_interp_params(16, 8, dt=DT)),
        'd0_16_16': (dirichlet_bc, zero_ic, build_interp_params(16, 16, dt=DT)),
        'd0_1_8': (dirichlet_bc, zero_ic, build_interp_params(1, 8, dt=DT)),
        'd0_2_8': (dirichlet_bc, zero_ic, build_interp_params(2, 8, dt=DT)),
        'd0_4_8': (dirichlet_bc, zero_ic, build_interp_params(4, 8, dt=DT)),
        'd0_8_8': (dirichlet_bc, zero_ic, build_interp_params(8, 8, dt=DT)),
    }

    with Pool(processes=N_PARALLEL) as p:
        res = []
        for key_item_pair in tasks.items():
            res.append(p.apply_async(run, key_item_pair))

        for r in res:
            print(r.get())

    dyn_vars = ['uz', 'ux', 'rho', 'P']
    dat = {name: strat_helper.load(args[0],
                                   dyn_vars=dyn_vars,
                                   name=name,
                                   **args[2])[2]
           for name, args in tasks.items()}
    plt.loglog(DT / np.array([0.5, 1, 2, 4, 8]),
               [rms_diff(dat['d0_dt0']['uz'], dat['d0_dt5']['uz']),
                rms_diff(dat['d0_16_8']['uz'], dat['d0_dt5']['uz']),
                rms_diff(dat['d0_dt2']['uz'], dat['d0_dt5']['uz']),
                rms_diff(dat['d0_dt3']['uz'], dat['d0_dt5']['uz']),
                rms_diff(dat['d0_dt4']['uz'], dat['d0_dt5']['uz'])], 'bo')
    plt.xlabel('Timestep')
    plt.ylabel('RMS difference w/ timestep %.3f' % (DT / 16))
    plt.title('3nd order Runge-Kutta timestep convergence in u_z')
    plt.savefig('t_conv.png')
    plt.clf()

    plt.semilogy(PARAMS_RAW['N_Z'] / np.array([16, 8, 4, 2]),
                 [rms_diff(dat['d0_16_16']['uz'], dat['d0_16_1']['uz']),
                  rms_diff(dat['d0_16_8']['uz'], dat['d0_16_1']['uz']),
                  rms_diff(dat['d0_16_4']['uz'], dat['d0_16_1']['uz']),
                  rms_diff(dat['d0_16_2']['uz'], dat['d0_16_1']['uz'])], 'bo')
    plt.xlabel('Number of z points')
    plt.ylabel('RMS difference from full %d z points' % PARAMS_RAW['N_Z'])
    plt.title('Convergence in uz, N_X held constant at %d' %
              (PARAMS_RAW['N_X'] / 16))
    plt.savefig('z_conv.png')
    plt.clf()

    plt.plot(PARAMS_RAW['N_X'] / np.array([16, 8, 4, 2]),
             [rms_diff(dat['d0_16_8']['uz'], dat['d0_1_8']['uz']),
              rms_diff(dat['d0_8_8']['uz'], dat['d0_1_8']['uz']),
              rms_diff(dat['d0_4_8']['uz'], dat['d0_1_8']['uz']),
              rms_diff(dat['d0_2_8']['uz'], dat['d0_1_8']['uz'])], 'bo')
    plt.xlabel('Number of x points')
    plt.ylabel('RMS difference from full/ %d x points' % PARAMS_RAW['N_X'])
    plt.title('Convergence in uz, N_Z held constant at %d' %
              (PARAMS_RAW['N_Z'] / 8))
    plt.savefig('x_conv.png')
    plt.clf()
