# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 

# ----------------------------------------------------------------------------- 
# Assignment 6
#
# Junjie Wei (jw4339)
# 
# Before: 1015846 function calls (1015725 primitive calls) in 7.314 seconds
# 
# After: 11834 function calls (11713 primitive calls) in 0.704 seconds
# 
# Speedup = 7.314 / 0.704 = 10.389
#


import numpy as np

def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = np.multiply(x,x)
    yy = np.multiply(y, y)
    zz = np.add(xx, yy)
    return np.sqrt(zz)