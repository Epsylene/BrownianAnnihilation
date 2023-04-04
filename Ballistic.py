import numpy as np
import matplotlib.pyplot as plt

class Ballistic:
    def __init__(self, n, c, L = 1, N = 100):
        self.n = n # number of particles
        self.c = c # initial proportion of particles

        self.N = N # number of time steps
        self.dt = 1/500 # time step interval

        self.L = L # box size
        v0 = 1 # particles initial velocity

        # Place particles at even intervals along the axis and
        # fill the rest with NaNs to avoid plotting values that
        # haven't been calculated
        self.x = np.ma.zeros((self.N, n))
        self.x[0] = self.L*np.random.choice(np.linspace(0.02, 0.98, n), size=n, replace=False)
        self.x[1:, :] = np.nan

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

        for ((t, i), _) in np.ndenumerate(x[0:N-1]):
            # If the particle has already been annihilated,
            # skip to the next one
            if (annihilated[t, i] == 1):
                continue

            # Collision checking: if the distance between two
            # particles changes sign between t and t+dt, then
            # they must have collided.
            for j in range(i):
                # Check that the particle hasn't already been
                # annihilated
                if(annihilated[t, j] == 1):
                    continue

                dx0 = x[t, i] - x[t, j]
                dx1 = x[t+1, i] - x[t+1, j]

                # If one is a particle and the other is an
                # anti-particle, then they annihilate each
                # other; else, they bounce off each other in an
                # elastic fashion.
                if abs(dx0) < 0.1 and dx0*dx1 < 0:
                    if particles[i] == -particles[j]:
                        # NaN values are not shown by
                        # matplotlib, so a particle and
                        # anti-particle tracks stop after their
                        # collision
                        x[t, i] = x[t, j] = np.nan
                        annihilated[t:, i] = annihilated[t:, j] = 1
                    else:
                        # Boucing off elastically means that
                        # velocities are reversed for both
                        # particles
                        v[i] *= -1; v[j] *= -1

            # Bounds checking: wrap particles around [0, L]
            if x[t, i] > L or x[t, i] < 0:
                x[t, i] %= L
                self.wrapped[t, i] = 1

            # Move all particles by one step
            x[t + 1] = x[t] + v*dt

    def plot(self):
        N, L = self.N, self.L

        # Avoid plotting lines between wrapped values
        x = np.ma.array(self.x, mask=self.wrapped)

        # Plot the trajectories of all particles
        for i, p in enumerate(self.particles):
            plt.plot(range(N), x[:, i], color = 'b' if p == 1 else 'r')
        
        # (t_wrap, i_wrap) = np.where(self.wrapped == 1)
        # plt.scatter(t_wrap, self.x[t_wrap, i_wrap], color='k', s=10)

        plt.xlim(-N/20, N + N/20)
        plt.ylim(-L/10, L + L/10)
        plt.show()