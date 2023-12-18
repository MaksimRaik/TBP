import numpy as np
from numba import njit

#@njit
def k( f, t, u, tau ):

    k = np.zeros( ( 4, u.size ) )

    k[ 0 ] = f( t, u )

    for i in np.arange( 1, 3, 1 ):

        k[ i ] = f( t + tau / 2., u + tau * k[ i - 1 ] / 2. )

    k[ -1 ] = f( t + tau, u + tau * k[ -2 ] )

    return k

#@njit
def RungeKutta( f, t0, u0, tend, tau ):

    global k

    t = np.arange( t0, tend, tau )

    u = np.zeros( ( t.shape[ 0 ], u0.size )  )

    u[ 0 ] = u0

    for i in np.arange( 1, t.shape[ 0 ], 1 ):

        kk = k( f, t[ i - 1 ], u[ i - 1 ], tau )

        C = tau / 6. * ( kk[ 0 ] + 2.0 * kk[ 1 ] + 2.0 * kk[ 2 ] + kk[ 3 ] )

        u[ i ] = u[ i - 1 ] + C

        #print( u[ i ] )

    return u, t