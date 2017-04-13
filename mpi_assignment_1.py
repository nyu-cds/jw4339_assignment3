'''
Advanced Python assignemtn 10:
    mpi_assignment_1.py
    
How to run:
    From prompt, mpiexec -n <num_of_tasks_to_invoke> python <path_to_this_script>
'''
from mpi4py import MPI

# names of MPI commands
comm = MPI.COMM_WORLD

# get rank 
rank = comm.Get_rank()

# determine if 
if rank % 2 == 0:
    # if even, print 'Hello'
    print('Hello from process {0}'.format(rank))
else:
    # if odd, print 'Goodbye'
    print('Goodbye from process {0}'.format(rank))
