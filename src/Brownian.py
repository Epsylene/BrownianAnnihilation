import numpy as np
import matplotlib.pyplot as plt

class Brownian:
    def __init__(self, n, c, N=100):
        self.n = n # number of particles
        self.c = c # initial proportion of particles
        self.N = N
        self.L = L = n

        # Place particles at even intervals along the axis and
        # fill the rest with NaNs to avoid plotting values that
        # haven't been calculated
        self.x = np.ma.zeros((self.N, n))
        self.x[0] = np.random.choice(np.linspace(-L, L, 2*n+1), size=n, replace=False)
        self.x[1:, :] = np.nan

        # Assign a number of particle and antiparticle state
        n_p = int(n*c)
        n_a = n - n_p
        self.particles = np.concatenate((-1*np.ones(n_a), 1*np.ones(n_p)))

        self.compute()

    def compute(self):
        N, n, L = self.N, self.n, self.L
        x, particles = self.x, self.particles

        self.annihilated = annihilated = np.zeros((N, n))
        dx = np.random.choice([-0.5, 0.5], size=(N, n))

        for (t, _) in enumerate(x[0:N-1]):
            # xt = np.ma.array(x[t], mask=annihilated[t])
            idx = np.argsort(x[t])
            _, u_pos = np.unique(x[t, idx], return_index=True)
            res = np.split(idx, u_pos[1:])
            res = list(filter(lambda x: x.size > 1, res))

            for e in res:
                sum = np.sum(particles[e])
                if sum == 0:
                    annihilated[t+1, e] = 1
                elif sum == 1:
                    annihilated[t+1:, e] = 1
                    e = e[np.where(particles[e] == 1)][0]
                    annihilated[t+1:, e] = 0
                elif sum == -1:
                    annihilated[t+1:, e] = 1
                    e = e[np.where(particles[e] == -1)][0]
                    annihilated[t+1:, e] = 0

            alive = np.where(annihilated[t+1] == 0)
            x[t+1, alive] = x[t, alive] + dx[t, alive]

    def plot(self, adaptative=False):
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