'''
Advanced Python assignemtn 10, task 2:
    mpi_assignment_2.py


Usage:
    mpiexec -n <num_of_tasks_to_invoke> python <path_to_this_script> [a_number]
        mpiexec                        : the command/program to invoke a MPI routine
        -n <num_of_tasks_to_invoke>    : invoke the MPI routine with specified number of tasks
        python                         : the python interpreter command
        <path_to_this_script>          : path to this script        
        [a_number]                     : if a number provided from the command line argument, the user won't be asked for intpu
                                         else the user will be ask for a input by the task with rank 0
'''

import sys
from mpi4py import MPI

# names of MPI commands
comm = MPI.COMM_WORLD

# get the size of processes
rank_size = comm.Get_size()

# get rank 
rank = comm.Get_rank()


# branching based on rank:
#    task with rank 0 will process user input; and first send then receive msg
#    tasks with rank other than 0 will just stand by, first receive then send msg
if rank == 0:
    # check if command line arguement exists
    if len(sys.argv) < 2:
        # if not provided from command line arguemnt, ask for an user input
        u_input = eval(input("Enter an arbitray number: "))
        while u_input >= 100:
            u_input = eval(input("The number should be less than 100, try another one: "))
    else:
        # if provided from command line argument
        u_input = eval(sys.argv[1])
        if u_input >= 100:
            print("The initial number should be less than 100, try run it again.\nExist...")
            quit()
        
    # start msg sending
    comm.send(u_input, dest = rank + 1)
    
    # receive msg
    res = comm.recv(source=rank_size-1)
    
    # print out result
    print('The result is {}'.format(res))
    
else:
    # check if the number is provided by command line arguement
    # if so, check the user input to determine if continue 
    if len(sys.argv) >= 2:
        u_input = eval(sys.argv[1])
        
        # check user input
        if u_input < 100:
            # tasks other than rank 0
            # receive msg first
            res = comm.recv(source=rank-1)
            
            # then send out
            comm.send(res * rank, dest = (rank + 1) % rank_size)
        else:
            pass
    else: 
    # if the number is not provided by command line input, just start listening msg
        res = comm.recv(source=rank-1)
        comm.send(res * rank, dest = (rank + 1) % rank_size)



