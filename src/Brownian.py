import numpy as np
import matplotlib.pyplot as plt

class Brownian:
    def __init__(self, n, c, N=100):
        self.n = n # number of particles
        self.c = c # initial proportion of particles
        self.N = N

        # Place particles at even intervals along the axis and
        # fill the rest with NaNs to avoid plotting values that
        # haven't been calculated
        self.x = np.ma.zeros((self.N, n))
        self.x[0] = np.random.choice(np.linspace(-1, 1, 2*n+1), size=n, replace=False)
        self.x[1:, :] = np.nan

        # Assign a number of particle and antiparticle state
        n_p = int(n*c)
        n_a = n - n_p
        self.particles = np.concatenate((-1*np.ones(n_a), 1*np.ones(n_p)))

        self.compute()

    def compute(self):
        N, n = self.N, self.n
        x, particles = self.x, self.particles

        self.annihilated = annihilated = np.zeros((N, n))
        dx = np.random.choice([-1/(2*n), 1/(2*n)], size=(N, n))

        for (t, _) in enumerate(x[0:N-1]):
            idx = np.argsort(np.round(x[t], decimals=6))
            # print(t, np.round(x[t], decimals=4))
            _, u_pos = np.unique(np.round(x[t, idx], decimals=6), return_index=True)
            res = np.split(idx, u_pos[1:])
            res = list(filter(lambda x: x.size > 1, res))

            for e in res:
                collided = particles[e] + 1
                if np.prod(collided) == 0 and ~np.all(collided == 0):
                    annihilated[t+1:, e] = 1

            alive = np.where(annihilated[t+1] == 0)
            x[t+1, alive] = x[t, alive] + dx[t, alive]

    def plot(self, adaptative=True):
        N, n = self.N, self.n

        if(adaptative): self.plot_N = 3*n
        else: self.plot_N = N

        # Plot the grid lines
        kwargs = {'color': 'k', 'ls': '--', 'lw': 0.2}
        for i in np.linspace(-1, 1, 2*n+1):
            plt.axline((0, i), slope=1/(2*n), **kwargs)
            plt.axline((0, i), slope=-1/(2*n), **kwargs)
        for i in np.arange(2, self.plot_N, 2):
            plt.axline((i, -1), slope=1/(2*n), **kwargs)
            plt.axline((i, 1), slope=-1/(2*n), **kwargs)

        # Mask the annihilated particles
        x = np.ma.array(self.x, mask=self.annihilated)

        # Plot the trajectories of all particles
        for i, p in enumerate(self.particles):
            plt.plot(range(N), x[:, i], color = 'b' if p == 1 else 'r')

        plt.ylim([-1, 1])
        plt.xlim([0, self.plot_N-1])
        plt.xticks([])
        plt.yticks([])

        plt.title('Discrete brownian walk of particles and antiparticles\n'+rf'($n_0 = {self.n}$, $c_0 = {self.c}$, $N = {N}$)')
        plt.xlabel('Time')
        plt.ylabel('Position')

        plt.show()