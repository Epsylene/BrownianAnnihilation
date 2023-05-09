import numpy as np
import matplotlib.pyplot as plt

class Ballistic:
    def __init__(self, n, c, N=100, L=1, v=('set', 1), bounded=True):
        '''
        Ballistic simulation constructor. The particles are
        placed uniformly along the axis of size L and given a
        random initial velocity. They are then given random
        particle/anti-particle states according to the
        concentration, and the `compute()` method running the
        simulation is called.

        Args:
            n: int
                The initial number of particles.
            c: float
                The initial concentration of particles, between
                0 and 1 (for example, c=0.3 means that there are
                30 particles out of a total of a hundred)
            L: float
                The size of the simulation box in which the
                particles live. Default 1.
            v: tuple
                A tuple containing a string (either 'set' or
                'gaussian') and a float v0. In the 'set' case,
                the initial velocity is drawn from the set {-v0,
                0, v0}; in the 'gaussian' case, it is a random
                value in a normal distribution with mean v0.
                Default ('set', 1).
            N: int
                Number of steps for the simulation. Default 100.
            bounded: bool
                Whether the simulation particles are bound
                inside the [0, L] box or not. Default True.
        '''
        self.n = n # number of particles
        self.c = c # initial proportion of particles
        self.N = N # number of time steps

        self.dt = 1/500 # time step interval
        self.L = L*(1 + n/500) # box size (made bigger as the 
            # number of particles increases to avoid floating
            # point issues when placing too much particles in a
            # box that is too small)

        vtype, v0 = v
        if vtype == 'set':
            self.v = np.random.choice([-v0, 0, v0], size=n)
        elif vtype == 'gaussian':
            self.v = np.random.normal(v0, size=n)

        # Place particles at even intervals along the axis and
        # fill the rest with NaNs to avoid plotting values that
        # haven't been calculated. The particles are not
        # initially set either on 0 or L to avoid issues with
        # bounds wrapping. The positions are stored as 32-bit
        # floats, which help reducing the memory usage in
        # compute() (and apparently, performance too ?)
        self.x = np.ma.ones((self.N, n), dtype=np.float32) * np.nan
        self.x[0] = np.random.choice(np.linspace(self.L*0.05, self.L*0.95, n), size=n, replace=False)

        a = int(2*n)
        self.space = np.zeros((N, a))
        self.space[:] = np.linspace(0, L, a)

        # Randomly assign a number of particle and anti-particle
        # states, according to the initial concentration
        n_p = int(n*c)
        n_a = n - n_p
        self.particles = np.concatenate((-1*np.ones(n_a), np.ones(n_p)))

        self.compute(bounded)

    def compute(self, bounded):
        '''
        Simulation runner. The distances between particles are
        computed at each time step and compared; if the sign
        changes between t and t+dt, the particles are supposed
        to have collided. If they are of opposite kind, they are
        both annihilated; if they are of the same kind, their
        velocities are swapped. Finally, all remaining particles
        are moved by v*dt and the loop starts a new iteration.
        '''
        n, L, N = self.n, self.L, self.N
        x, v, particles, dt = self.x, self.v, self.particles, self.dt

        self.annihilated = annihilated = np.zeros((N, n))
        self.wrapped = np.zeros((N, n))

        for (t, _) in enumerate(x[1:N-1]):
            # Build two matrices from the outer substraction of
            # the positions at t and t-1 ; that is, the element
            # (i, j) of each of these matrices is given by (xi -
            # xj) at the corresponding time. Multiply then these
            # matrices element-wise: in the resulting matrix,
            # each element corresponds to the product
            # d_ij(t-1)*d_ij(t). If this product is negative, it
            # means that the distance between the particles has been
            # reversed between t and t-1 ; if on top of that the two
            # particles were close to start with (abs(diff_t) < 0.1),
            # then they must have collided between t-1 and t.
            diff_t = np.subtract.outer(x[t-1], x[t-1])
            diff_t1 = np.subtract.outer(x[t], x[t])
            cross_diff = np.multiply(diff_t, diff_t1)
            coll = np.argwhere(np.logical_and(abs(diff_t) < 0.1, cross_diff < 0))
            
            for e in coll:
                # For each pair of particles that have met,
                # check if they are of opposite kind (in which
                # case the sum of the particle states will be
                # zero).
                if np.sum(particles[e]) == 0:
                    annihilated[t:, e] = 1
                    x[t, e] = np.nan
                # If they are of the same kind, simply swap
                # their velocities.
                else:
                    a, b = e
                    v[[a, b]] = v[[b, a]]

            # Bounds checking: all particles that are outside of
            # the box are wrapped around to the other side.
            if bounded:
                out = np.where(np.logical_or(x[t] > L, x[t] < 0))
                x[t, out] %= L
                self.wrapped[t, out] = 1

            # Move all particles by one step
            x[t + 1] = x[t] + v*dt

    def plot(self, savefig=False):
        '''
        Plot the trajectories of the particles as time-position
        lines. Particle trajectories are colored in blue, while
        antiparticles are in red. 
        '''
        N, L = self.N, self.L

        # Avoid plotting lines between wrapped values
        x = np.ma.array(self.x, mask=self.wrapped)

        # Plot the trajectories of all particles
        for i, p in enumerate(self.particles):
            plt.plot(range(N), x[:, i], color = 'b' if p == 1 else 'r')

        plt.xlim(-N/20, N + N/20)
        plt.ylim(-L*0.05, L*1.05)
        plt.xticks([])
        plt.yticks([])

        plt.title('Ballistic simulation of particles and antiparticles\n'+rf'($n_0 = {self.n}$, $c_0 = {self.c}$, $N = {N}$)')
        plt.xlabel('Time')
        plt.ylabel('Position')

        if savefig: plt.savefig(f'plot_ball_n{self.n}_N{N}_c{self.c}.pdf', bbox_inches='tight')
        plt.show()