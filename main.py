import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation, PillowWriter
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

mu = 1. / 82.45

alpha = mu / ( 1.0 + mu )

betta = ( 1.0 - mu ) / ( 1.0 + mu )

xL = np.asarray( [ ( 1. - ( alpha / 3. ) ), ( 1. + ( alpha / 3. ) ), - ( 1 + 5. / 12. * alpha ), 0.5 - betta, 0.5 - betta ] )

yL = np.asarray( [ 0.0, 0.0, 0.0, np.sqrt( 3. ) / 2., - np.sqrt( 3. ) / 2. ] )

#u0 = np.asarray( [ 1.2, -1.05, 0.0, -1.05 ] )

u0 = np.asarray( [ yL[ 1 ], xL[ 1 ], 0.0, 0.0 ] )

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

def init():
	# creating an empty plot/frame
	line2.set_data([], [])
	return line2,

def update(frame):
    #for each frame, update the data stored on each artist.
    x = uRK[:frame,0]
    y = uRK[:frame,1]
    # update the scatter plot:
    data = np.stack([x, y]).T
    # update the line plot:
    line2.set_data( uRK[:frame,0], uRK[:frame,1] )
    #line2.set_xdata()
    #line2.set_ydata()
    return line2,

ani = FuncAnimation(fig=fig, func=update, init_func=init, frames=tRK.size, interval=1, blit=True)
fig.suptitle(f'$x_0$ = {round(u0[0],3)}; $y_0$ = {round(u0[1],3)}; $k$ = {kk}', fontsize=14)
ani.save(f'trak x_0 = {round(u0[0],3)}; y_0 = {round(u0[1],3)}; k = {kk} 2.gif', fps=30)
plt.close()
#plt.show()


