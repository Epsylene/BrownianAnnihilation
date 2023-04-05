import numpy as np
import matplotlib.pyplot as plt

def concentration(simul, full_plot=False):
        '''
        Calculate the concentration of particles in the given
        simulation over time.

        Args:
            simul: Brownian, Ballistic
                The simulation.
            full_plot: bool
                If True, plot the concentration between 0 and 1,
                regardless of its starting value. If False, plot
                between the minimum and maximum values.
        '''
        N = simul.N

        # Indices of the particles
        p_idx = np.where(simul.particles == 1)[0]
        x = np.ma.array(simul.x, mask=simul.annihilated)

        # The concentration of particles over time is the number
        # particles divided by the total number of particles and
        # antiparticles.
        concentration = np.zeros(N)
        for t in range(N):
            concentration[t] = np.ma.count(x[t, p_idx])/np.ma.count(x[t])

        plt.plot(range(N), concentration)
        if full_plot: plt.ylim(0, 1)

def left(simul, params, n_exp):
    '''
    Plots the average concentrations for a given type of
    simulation over a number of experiments with different
    parameters.

    Args:
        simul: Brownian, Ballistic
            The type of simulation to perform.
        params: array with 2 tuples
            The simulations parameters. The first tuple is
            expected to contain the initial numbers of particles
            and the second the initial concentrations.
        n_exp: int
            The number of experiments to perform for each pair
            of parameters.
    '''
    n, c = params
    concentrations = np.zeros((len(c), len(n), n_exp))

    for (i, c0) in enumerate(c):
        for (j, n0) in enumerate(n):
            for k in range(n_exp):
                b = simul(n0, c0)
                all = np.ma.array(b.particles, mask=b.annihilated[b.N - 1])
                particles = np.where(all == 1)
                concentrations[i, j, k] = np.size(particles)/np.ma.count(all)
    
    init_pop = [rf'$n_0 = {n0}$' for n0 in n]
    avg_c = [[np.average(c_n) for c_n in c] for c in concentrations]

    x = np.arange(len(init_pop))
    for (i, c0) in enumerate(c):
        plt.bar(x + 0.3*i, avg_c[i], width=0.3, label=rf'$c_0$ = {c0}')
    plt.xticks(x + 0.15, init_pop)
    plt.title(f'Average final concentration over {n_exp} experiments\nwith different initial populations and concentrations')
    plt.legend()

    plt.show()