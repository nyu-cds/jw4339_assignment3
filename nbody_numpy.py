"""
    N-body simulation.
    
    Module: nbody_iter.py
    Author: jw4339@nyu.edu
    
    Assignment 6
"""

"""
cProfile result [Before]
         306 function calls in 89.907 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.000    0.000 nbody_iter.py:100(offset_momentum)
        1    0.019    0.019   89.891   89.891 nbody_iter.py:117(nbody)
      100   89.816    0.898   89.870    0.899 nbody_iter.py:52(advance)
      100    0.002    0.000    0.002    0.000 nbody_iter.py:81(report_energy)
        1    0.016    0.016   89.907   89.907 nbody_iter.py:9(<module>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 {method 'keys' of 'dict' objects}
      101    0.055    0.001    0.055    0.001 {range}
"""

"""
cProfile result [After]

"""


import itertools
import numpy as np

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

BODIES = {
    'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),

    'jupiter': ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
                9.54791938424326609e-04 * SOLAR_MASS),

    'saturn': ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR],
               2.85885980666130812e-04 * SOLAR_MASS),

    'uranus': ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR],
               4.36624404335156298e-05 * SOLAR_MASS),

    'neptune': ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
                5.15138902046611451e-05 * SOLAR_MASS)}


                
                
                
# def advance(dt, iterations, all_combinations, local_bodies_dict):
#     '''
#         advance the system iterations timesteps, dt timestep each
#     '''
#     for _ in range(iterations):
#         for (body1, body2) in all_combinations:
#             ([x1, y1, z1], v1, m1) = local_bodies_dict[body1]
#             ([x2, y2, z2], v2, m2) = local_bodies_dict[body2]
#             # comput deltas
#             (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
#             # update v's
#             mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
#             factor1 = m1 * mag
#             factor2 = m2 * mag
#             v1[0] -= dx * factor2
#             v1[1] -= dy * factor2
#             v1[2] -= dz * factor2
#             v2[0] += dx * factor1
#             v2[1] += dy * factor1
#             v2[2] += dz * factor1
#             
#         #####
#         # TODO: numpy optimization
#         ######
#         for body in local_bodies_dict:
#             (r, [vx, vy, vz], m) = local_bodies_dict[body]
#             # update r's
#             r[0] += dt * vx
#             r[1] += dt * vy
#             r[2] += dt * vz

@profile
def advance(dt, iterations, all_combinations, bodies_ndarray):
    '''
        advance the system iterations timesteps, dt timestep each
    '''
    for _ in range(iterations):
        for body_idx_1, body_idx_2 in all_combinations:
            #([x1, y1, z1], v1, m1) = local_bodies_dict[body1]
            #([x2, y2, z2], v2, m2) = local_bodies_dict[body2]
            
            # comput deltas
            # (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
            deltas = bodies_ndarray[body_idx_1, 0:3] - bodies_ndarray[body_idx_2, 0:3]
            
            
            # update v's
            # mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            # factor1 = m1 * mag
            # factor2 = m2 * mag
            # v1[0] -= dx * factor2
            # v1[1] -= dy * factor2
            # v1[2] -= dz * factor2
            # v2[0] += dx * factor1
            # v2[1] += dy * factor1
            # v2[2] += dz * factor1
            mag = dt * (np.sum(deltas ** 2) ** (-1.5))
            bodies_ndarray[body_idx_1, 3:6] -= deltas * mag * bodies_ndarray[body_idx_2, -1:]
            bodies_ndarray[body_idx_2, 3:6] += deltas * mag * bodies_ndarray[body_idx_1, -1:]
            
        #####
        # TODO: numpy optimization
        ######
        # for body in local_bodies_dict:
        #     (r, [vx, vy, vz], m) = local_bodies_dict[body]
        #     # update r's
        #     r[0] += dt * vx
        #     r[1] += dt * vy
        #     r[2] += dt * vz 
        
        # update r's
        bodies_ndarray[:, 0:3] += dt * bodies_ndarray[:, 3:6]

def report_energy(all_combinations, bodies_ndarray, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    # for (body1, body2) in all_combinations:
    #     ((x1, y1, z1), v1, m1) = local_bodies_dict[body1]
    #     ((x2, y2, z2), v2, m2) = local_bodies_dict[body2]
    #     # compute deltas
    #     (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
    #     # compute energy
    #     e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
    
    for body_idx_1, body_idx_2 in all_combinations:
        # compute deltas
        deltas = bodies_ndarray[body_idx_1, 0:3] - bodies_ndarray[body_idx_2, 0:3]
        # compute energy
        e -= bodies_ndarray[body_idx_1, -1] * bodies_ndarray[body_idx_2, -1] / (np.sum(deltas ** 2) ** 0.5)
    
    
    #####
    # TODO: numpy optimization
    ######
    # for body in local_bodies_dict:
    #     (r, [vx, vy, vz], m) = local_bodies_dict[body]
    #     e += m * (vx * vx + vy * vy + vz * vz) / 2.
    
    for i in range(bodies_ndarray.shape[0]):
        e += bodies_ndarray[i, -1] * np.sum(bodies_ndarray[i, 3:6] ** 2) / 2.0
    
    return e

# def offset_momentum(ref, local_bodies_dict, px=0.0, py=0.0, pz=0.0):
#     '''
#         ref is the body in the center of the system
#         offset values from this reference
#     '''
#     
#     #####
#     # TODO: numpy optimization
#     ######    
#     for body in local_bodies_dict:
#         (r, [vx, vy, vz], m) = local_bodies_dict[body]
#         px -= vx * m
#         py -= vy * m
#         pz -= vz * m
#         
#     (r, v, m) = ref
#     v[0] = px / m
#     v[1] = py / m
#     v[2] = pz / m

@profile  
def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # init bodies data using numpy ndarray
    sun = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS])
    jupiter = np.array([4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01, 
                        1.66007664274403694e-03, 7.69901118419740425e-03, -6.90460016972063023e-05, 
                        9.54791938424326609e-04])
    saturn = np.array([8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01, 
                       -2.76742510726862411e-03, 4.99852801234917238e-03, 2.30417297573763929e-05, 
                       2.85885980666130812e-04])
    uranus = np.array([1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01, 
                       2.96460137564761618e-03, 2.37847173959480950e-03, -2.96589568540237556e-05, 
                       4.36624404335156298e-05]) 
    neptune = np.array([1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01, 
                        2.68067772490389322e-03, 1.62824170038242295e-03, -9.51592254519715870e-05, 
                        5.15138902046611451e-05])    
    bodies_ndarray = np.vstack((sun, jupiter, saturn, uranus, neptune))
    bodies_ndarray[1:, 3:6] *= DAYS_PER_YEAR
    bodies_ndarray[1:, -1:] *= SOLAR_MASS
    
    
    # local variable
    # local_bodies_dict = BODIES
    
    # Set up global state
    # offset_momentum(BODIES[reference], local_bodies_dict)
    bodies_ndarray[0, 3:6] -= ((bodies_ndarray[:, 3:6].T * bodies_ndarray[:, -1]).T).sum(axis=0)
    bodies_ndarray[0, 3:6] /= bodies_ndarray[0, -1]
    

    # get all combination of body keys
    # all_combinations = list(itertools.combinations(BODIES.keys(), 2))
    all_combinations = list(itertools.combinations(range(0, 5), 2))
    
    for _ in range(loops):
        # advance(0.01, iterations, all_combinations, local_bodies_dict)
        # print(report_energy(all_combinations, local_bodies_dict))
        advance(0.01, iterations, all_combinations, bodies_ndarray)
        print(report_energy(all_combinations, bodies_ndarray))
        
if __name__ == '__main__':
    # nbody(100, 'sun', 20000)
    nbody(5, 'sun', 20000)

