import numpy as np
import matplotlib.pyplot as plt

def concentration(simul, full_plot=False):
        N = simul.N

        # Indices of the particles
        p_idx = np.where(simul.particles == 1)[0]
        x = np.ma.array(simul.x, mask=simul.annihilated)

        concentration = np.zeros(N)
        for t in range(N):
            concentration[t] = np.ma.count(x[t, p_idx])/np.ma.count(x[t])

        plt.plot(range(N), concentration)
        if full_plot: plt.ylim(0, 1)

def left(simul):
    all = simul.particles[simul.annihilated == 0]
    left_particles = np.size(np.where(all == 1))
    left_all = np.size(all)

    return left_particles/left_all