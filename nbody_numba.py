"""
    N-body simulation.
    
    Module: nbody_numba.py
    Author: Junjie Wei (jw4339@nyu.edu)
    
"""
import itertools
import numpy as np
from numba import jit

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

BODIES = {
    'sun': (np.array([0.0, 0.0, 0.0], dtype=np.float64), 
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            SOLAR_MASS),

    'jupiter': (np.array([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01], dtype=np.float64),
                np.array([1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR], dtype=np.float64),
                9.54791938424326609e-04 * SOLAR_MASS),

    'saturn': (np.array([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01], dtype=np.float64),
               np.array([-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR], dtype=np.float64),
               2.85885980666130812e-04 * SOLAR_MASS),

    'uranus': (np.array([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01], dtype=np.float64),
               np.array([2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR], dtype=np.float64),
               4.36624404335156298e-05 * SOLAR_MASS),

    'neptune': (np.array([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01], dtype=np.float64),
                np.array([2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR], dtype=np.float64),
                5.15138902046611451e-05 * SOLAR_MASS)}

@jit
def vec_deltas(a, b):
    return a - b
                
                
@jit
def advance(dt, iterations, all_combinations, local_bodies_dict):
    '''
        advance the system iterations timesteps, dt timestep each
    '''
    local_bodies_dict = BODIES
    body_keys = local_bodies_dict.keys()
    for _ in range(iterations):
        for (body1, body2) in all_combinations:
            (pos1, v1, m1) = local_bodies_dict[body1]
            (pos2, v2, m2) = local_bodies_dict[body2]
            # comput deltas
            delta_pos = vec_deltas(pos1, pos2)
            # update v's
            mag = dt * (np.sum(delta_pos**2) ** (-1.5))
            factor1 = m1 * mag
            factor2 = m2 * mag
            v1 -= delta_pos * factor2
            v2 += delta_pos * factor1
            
        for body in body_keys:
            (r, v, m) = local_bodies_dict[body]
            # update r's
            r += dt * v

@jit
def report_energy(all_combinations, local_bodies_dict, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    local_bodies_dict = BODIES
    body_keys = local_bodies_dict.keys()

    for (body1, body2) in all_combinations:
        (pos1, v1, m1) = local_bodies_dict[body1]
        (pos2, v2, m2) = local_bodies_dict[body2]
        # comput deltas
        delta_pos = vec_deltas(pos1, pos2)
        # comput energy
        e -= (m1 * m2) / (np.sum(delta_pos**2) ** 0.5)
        
    for body in body_keys:
        (r, v, m) = local_bodies_dict[body]
        e += m * np.sum(v**2) / 2.
        
    return e

@jit
def offset_momentum(ref, local_bodies_dict, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    # local_bodies_dict = BODIES
    body_keys = local_bodies_dict.keys()
    
    for body in body_keys:
        (r, v, m) = local_bodies_dict[body]
        px -= v[0] * m
        py -= v[1] * m
        pz -= v[2] * m
        
    (r, v, m) = ref
    v = np.array([px / m, py / m, pz / m])


@jit
def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # local variable
    local_bodies_dict = BODIES
    
    # Set up global state
    offset_momentum(local_bodies_dict[reference], local_bodies_dict)

    # get all combination of body keys
    all_combinations = list(itertools.combinations(local_bodies_dict.keys(), 2))
    
    for _ in range(loops):
        advance(0.01, iterations, all_combinations, local_bodies_dict)
        print(report_energy(all_combinations, local_bodies_dict))

if __name__ == '__main__':
    nbody(100, 'sun', 20000)

