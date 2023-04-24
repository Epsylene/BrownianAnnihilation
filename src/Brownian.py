import numpy as np
import matplotlib.pyplot as plt

class Brownian:
    def __init__(self, n, c, N=100):
        '''
        Brownian simulation constructor. The particles are set
        at random places along an axis with 2n+1 uniformly
        distributed positions, and then given random
        particle/anti-particle states according to the
        concentration. The `compute()` method, running the
        simulation, is then called.

        Args:
            n: int
                The initial number of particles.
            c: float
                The initial concentration of particles, between
                0 and 1 (for example, c=0.3 means that there are
                30 particles out of a total of a hundred)
            N: int
                Number of steps for the simulation. Default 100.
        '''
        self.n = n # number of particles
        self.c = c # initial proportion of particles
        self.N = N
        self.L = L = n

        # Place particles at even intervals along the axis and
        # fill the rest with NaNs to avoid plotting values that
        # haven't been calculated
        self.x = np.ma.ones((self.N, n))*np.nan
        self.x[0] = np.random.choice(np.linspace(-L, L, 2*n+1), size=n, replace=False)

        # Assign a number of particle and antiparticle state
        n_p = int(n*c)
        n_a = n - n_p
        self.particles = np.concatenate((-1*np.ones(n_a), 1*np.ones(n_p)))

        self.compute()

    def compute(self):
        '''
        Simulation runner. At each time step, the positions at
        which lie the particles are checked to see if there two
        are more on the same spot; if this is found to be the
        case (that is, if the particles have met), all pairs
        particle/anti-particle are destroyed. All the remaining
        particles are randomly moved either up or down on the
        simulation lattice, and the loop starts a new iteration.
        '''
        N, n, L = self.N, self.n, self.L
        x, particles = self.x, self.particles

        # The random moves of particles are precomputed for
        # performance
        self.annihilated = annihilated = np.zeros((N, n))
        dx = np.random.choice([-0.5, 0.5], size=(N, n))

        for (t, _) in enumerate(x[0:N-1]):
            # The positions of the particles are sorted and
            # separated in groups of unique values; all groups
            # except the ones containing more than one element
            # (which correspond to the spots containing more
            # than one particle) are then filtered out.
            sorted_idx = np.argsort(x[t])
            _, u_idx = np.unique(x[t, sorted_idx], return_index=True)
            res = np.split(sorted_idx, u_idx[1:])
            res = list(filter(lambda x: x.size > 1, res))

            # For each of these groups of particles-packed
            # positions, the global sum of the particle states
            # is calculated: if it is 0 (as much particles as
            # antiparticles), all elements in the group are
            # annihilated. If it is 1 or -1, all particles are
            # destroyed except the remaining particle or
            # anti-particle.
            for e in res:
                sum = np.sum(particles[e])
                if sum == 0:
                    annihilated[t+1:, e] = 1
                elif sum == 1:
                    annihilated[t+1:, e] = 1
                    x[t, e] = np.nan
                    e = e[np.where(particles[e] == 1)][0]
                    annihilated[t+1:, e] = 0
                elif sum == -1:
                    annihilated[t+1:, e] = 1
                    e = e[np.where(particles[e] == -1)][0]
                    annihilated[t+1:, e] = 0

            # All remaining particles are then moved by dx[t]
            alive = (annihilated[t+1] == 0)
            x[t+1, alive] = x[t, alive] + dx[t, alive]

    def plot(self, adaptative=False):
        '''
        Plot the trajectories of the particles as time-position
        lines, drawn on a grid that represents the simulation
        lattice. Particle trajectories are colored in blue,
        while antiparticles are in red. Grid lines are not drawn
        for any n>=100 unless `adaptative` is set to True, to
        avoid cluttering of the plot.

        Args:
            adaptative: boolean
                If True, change the figure size from the default
                to any size necessary to make a compact plot of
                all the points in the simulation while
                maintaining a nice aspect ratio. Be careful that
                this can produce very large figsizes for large
                values of n or N.
        '''
        N, n, L = self.N, self.n, self.L
        plot_N = N if adaptative else 3*n
        
        if adaptative: plt.figure(figsize=(7/100*N, 1/5*n))
        
        # Plot the grid lines
        kwargs = {'color': 'k', 'ls': '--', 'lw': 0.2}
        if n < 100 or adaptative:
            for i in np.linspace(-L, L, 2*n+1):
                plt.axline((0, i), slope=1/2, **kwargs)
                plt.axline((0, i), slope=-1/2, **kwargs)
            for i in np.arange(2, plot_N, 2):
                plt.axline((i, -L), slope=1/2, **kwargs)
                plt.axline((i, L), slope=-1/2, **kwargs)

        # Mask the annihilated particles
        x = np.ma.array(self.x, mask=self.annihilated)

        # Plot the trajectories of all particles
        for i, p in enumerate(self.particles):
            plt.plot(range(N), x[:, i], color = 'b' if p == 1 else 'r')

        plt.ylim([-L, L])
        plt.xlim([0, N-1 if adaptative else 3*n])
        plt.xticks([])
        plt.yticks([])

        plt.title('Discrete brownian walk of particles and antiparticles\n'+rf'($n_0 = {self.n}$, $c_0 = {self.c}$, $N = {N}$)')
        plt.xlabel('Time')
        plt.ylabel('Position')

        plt.show()