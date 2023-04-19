import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scp

def concentration(simul, plot=False):
        '''
        Return and eventually plot the concentration of
        particles in the given simulation over time.

        Args:
            simul: Brownian, Ballistic
                The simulation.
            plot: bool
                If True, plot the concentration between its
                minimum and maximum values.
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

        if plot:
            plt.plot(range(N), concentration)

            plt.title(rf'Concentration of particles over time ($n_0 = {simul.n}$, $c_0 = {simul.c}$)')
            plt.xlabel('Time')
            plt.ylabel('Concentration')
            plt.show()

        return concentration

def end_state(simul, params, n_exp):
    '''
    Plots the average final concentrations for a given type of
    simulation over a number of experiments with different
    parameters.

    Args:
        simul: Brownian, Ballistic
            The type of simulation to perform.
        params: array with 2 tuples
            The simulations parameters. The first tuple is
            expected to contain the initial numbers of particles
            and the second the initial concentrations. An
            example input is [(5, 10, 20), (0.2, 0.8)].
        n_exp: int
            The number of experiments to perform for each pair
            of parameters.
    '''
    n, c = params
    concentrations = np.zeros((len(c), len(n), n_exp))

    # For each pair (n0, c0) (initial number of particles and
    # concentration), repeatedly create a simulation and
    # calculate the end concentration (non-annihilated particles
    # over all non-annihilated)
    for (i, c0) in enumerate(c):
        for (j, n0) in enumerate(n):
            for k in range(n_exp):
                b = simul(n0, c0)
                all = np.ma.array(b.particles, mask=b.annihilated[b.N - 1])
                particles = np.where(all == 1)
                concentrations[i, j, k] = np.size(particles)/np.ma.count(all)
    
    # Average the concentrations over 
    n0_labels = [rf'$n_0 = {n0}$' for n0 in n]
    avg_c = [[np.average(c_n) for c_n in c] for c in concentrations]

    # Display a bar chart of the 
    x = np.arange(len(n0_labels))
    for (i, c0) in enumerate(c):
        plt.bar(x + 0.3*i, avg_c[i], width=0.3, label=rf'$c_0$ = {c0}')
    plt.xticks(x + 0.15, n0_labels)
    plt.title(f'Average final concentration over {n_exp} experiments\nwith different initial populations and concentrations')
    plt.legend()

    plt.show()

def avg_concentration(simul, params, n_exp):

    n0, c0, *_ = params
    N = _[0] if _ else 100

    t = np.arange(N)
    c = np.zeros((n_exp, N))
    for i in range(n_exp):
        b = simul(*params)
        c[i] = concentration(b)
        plt.plot(t, c[i], lw=0.5)

    c = np.average(c, axis=0)
    plt.plot(c, c='k')

    plt.title(f'Average concentration over {n_exp}\n'+rf'experiments with $n_0 = {n0}$, $c_0 = {c0}$')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.show()

    return c

def fit(concentration):
    t = np.arange(len(concentration))
    c_model = lambda t, a, b, c, d: a + b/(t+c)**d

    args, _ = scp.curve_fit(c_model, t, concentration)
    fit = c_model(t, *args)
    r2 = 1 - np.sum((concentration - fit)**2)/np.sum((concentration - np.average(concentration))**2)

    print('Concentration fit a+b/(t+c)^d, with:')
    print('a = {}, b = {},\nc = {}, d = {}'.format(*args))
    print(f'R^2 = {r2}')

    plt.scatter(t, concentration, c='k', s=1, label='Simulation')
    plt.plot(t, fit, label='Model')

    s = '-' if args[1] < 0 else ''
    plt.title(rf'Concentration fit, {s}$1/t^\alpha$ with $\alpha = {args[3]:.2f}$')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()

    plt.show()