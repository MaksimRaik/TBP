import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from icecream import ic
from RungeKutta import *

def Calc( u0, mu, kk, tBEG, tEND, tau ):

    def f( t, u ):

        f_out = np.zeros( u.size )

        r1 = np.sqrt((u[0] + mu) ** 2 + u[1] ** 2)

        r2 = np.sqrt((u[0] - ( 1. - mu ) ) ** 2 + u[1] ** 2)

        f_out[ 0 ] = u[ 2 ]

        f_out[ 1 ] = u[ 3 ]

        f_out[ 2 ] = 2. * u[ 3 ] + u[ 0 ] - ( 1. - mu ) * ( u[ 0 ] + mu ) / r1 ** 3 - mu * ( u[ 0 ] - ( 1. - mu ) ) / r2 ** 3 - kk * u[ 2 ]

        f_out[ 3 ] = -2. * u[ 2 ] + u[ 1 ] - ( 1. - mu ) * u[ 1 ] / r1 ** 3 - mu * u[ 1 ] / r2 ** 3 - kk * u[ 3 ]

        return f_out

    uRK, tRK = RungeKutta( f, tBEG, u0, tEND, tau )

    return tRK, uRK

tbeg = 0.

tend = 200.

tau = 0.05

kk = 0.0

mu = 0.001#1. / 82.45

alpha = mu / ( 1.0 + mu )

betta = ( 1.0 - mu ) / ( 1.0 + mu )

xL = np.asarray( [ ( 1. - ( alpha / 3. ) ** 3 ), ( 1. + ( alpha / 3. ) ** 3 ), - ( 1 + 5. / 12. * alpha ), 0.5 - mu, betta / 2. ] )

yL = np.asarray( [ 0.0, 0.0, 0.0, np.sqrt( 3. ) / 2., - np.sqrt( 3. ) / 2. ] )

#u0 = np.asarray( [ 1.2, -1.05, 0.0, -1.05 ] )

u0 = np.asarray( [ yL[ 3 ], yL[ 3 ], 0.0, 0.0 ] )

#u0 = np.asarray( [ 0.94, -2.03, -1.05, 0.0 ] )

tRK, uRK = Calc( u0, mu, kk, tbeg, tend, tau )

print( uRK )

#plt.figure( figsize = ( 15, 10 ) )
#plt.plot( uRK[ :,0 ], uRK[ :,1 ], 'bo' )
#plt.grid()
#plt.show()

fig, ax = plt.subplots()

line2 = ax.plot(uRK[:,0], uRK[:,1], label='tract')[0]
#ax.plot( [ 1 - mu, ], [ 0, ], 'ro', label = 'Земля' )
#ax.plot( [ -mu, ], [ 0, ], 'go', label = 'Луна' )
#ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
ax.legend()

def update(frame):
    #for each frame, update the data stored on each artist.
    x = uRK[:frame,0]
    y = uRK[:frame,1]
    # update the scatter plot:
    data = np.stack([x, y]).T
    # update the line plot:
    line2.set_xdata(uRK[:frame,0])
    line2.set_ydata(uRK[:frame,1])
    return line2

ani = animation.FuncAnimation(fig=fig, func=update, frames=tRK.size, interval=1)
plt.show()



