import numpy as np
import matplotlib.pyplot as plt

class Ballistic:
    def __init__(self, n, c, L = 1, v0 = 1, N = 100):
        self.n = n # number of particles
        self.c = c # initial proportion of particles
        self.N = N # number of time steps

        self.dt = 1/500 # time step interval
        self.L = L*(1 + n/500) # box size
        self.v0 = v0 # particles initial velocity

        # Place particles at even intervals along the axis and
        # fill the rest with NaNs to avoid plotting values that
        # haven't been calculated
        self.x = np.ma.ones((self.N, n), dtype=np.float32) * np.nan
        self.x[0] = np.random.choice(np.linspace(self.L*0.05, self.L*0.95, n), size=n, replace=False)

        # Randomly assign a number of particle and anti-particle
        # states, according to the initial concentration
        n_p = int(n*c)
        n_a = n - n_p
        self.particles = np.concatenate((-1*np.ones(n_a), np.ones(n_p)))

        # Give random initial velocities to the particles:
        # either -v0, 0 or v0.
        self.v = np.random.choice([-v0, 0, v0], size=n)

        self.compute()

    def compute(self):
        n, L, N = self.n, self.L, self.N
        x, v, particles, dt = self.x, self.v, self.particles, self.dt

        self.annihilated = annihilated = np.zeros((N, n))
        self.wrapped = np.zeros((N, n))

        for (t, _) in enumerate(x[1:N-1]):
            diff_t = np.subtract.outer(x[t-1], x[t-1])
            diff_t1 = np.subtract.outer(x[t], x[t])
            diff = np.multiply(diff_t, diff_t1)
            coll = np.argwhere(np.logical_and(abs(diff_t) < 0.1, diff < 0))
            
            for e in coll:
                if np.sum(particles[e]) == 0:
                    annihilated[t:, e] = 1
                    x[t, e] = np.nan
                else:
                    a, b = e
                    v[[a, b]] = v[[b, a]]

            out = np.where(np.logical_or(x[t] > L, x[t] < 0))
            x[t, out] %= L
            self.wrapped[t, out] = 1

            # Move all particles by one step
            x[t + 1] = x[t] + v*dt

    def plot(self):
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

        plt.show()