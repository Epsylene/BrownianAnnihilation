import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scp

def concentration(simul, plot=False):
        '''
        Return and eventually plot the concentration (as in
        density) of particles in the given simulation over time.

        Args:
            simul: Brownian, Ballistic
                The simulation.
            plot: bool
                If True, plot the concentration between its
                minimum and maximum values. Default False.
        '''
        N = simul.N

        # Indices of the particles
        p_idx = np.where(simul.particles == 1)[0]
        x = np.ma.array(simul.x, mask=simul.annihilated)

        # The concentration of particles over time is the number
        # particles divided by the size of the box.
        density = np.zeros(N)
        for t in range(N):
            density[t] = np.ma.count(x[t, p_idx])/simul.L

        if plot:
            plt.plot(range(N), density)

            plt.title(rf'Density of particles over time ($n_0 = {simul.n}$, $c_0 = {simul.c}$)')
            plt.xlabel('Time')
            plt.ylabel('Concentration')
            plt.show()

        return density

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

    n0, c0, *targs = params
    N = targs[0] if targs else 100

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

def fit(concentration, recursion=1000):
    t = np.arange(len(concentration))
    c_model = lambda t, a, b, c, d: a + b/(t+c)**d

    args, _ = scp.curve_fit(c_model, t, concentration, maxfev=recursion)
    fit = c_model(t, *args)
    r2 = 1 - np.sum((concentration - fit)**2)/np.sum((concentration - np.average(concentration))**2)

    print('Concentration fit a+b/(t+c)^d, with:')
    print('a = {}, b = {},\nc = {}, d = {}'.format(*args))
    print(f'R^2 = {r2}')

    plt.scatter(t, concentration, c='k', s=1, label='Simulation')
    plt.plot(t, fit, label='Fit')

    s = '-' if args[1] < 0 else ''
    plt.title(rf'Concentration fit, {s}$1/t^\alpha$ with $\alpha = {args[3]:.2f}$')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()

    plt.show()

    return args[3]

def distribution(simul, m=None, T=None, proj='2d', savefig=False):
    n, N, L = simul.n, simul.N, simul.L
    if T == None: T = simul.N-2
    if m == None: m = 50 if proj == '2d' else 10

    x = simul.space
    distr = np.zeros(x.shape)
    part = simul.particles

    for t in range(T+1):
        for (i, xi) in enumerate(simul.x[t]):
            idx = np.where(abs(x[t] - xi) < 0.001)
            distr[t, idx] = part[i]

    if proj == '2d':
        plt.scatter(np.nan, np.nan, c='b', s=1, label='Particles')
        plt.scatter(np.nan, np.nan, c='r', s=1, label='Anti-particles')
        for (i, d) in enumerate(distr[T]):
            l = {1: 'b', -1: 'r'}
            if d != 0:
                plt.scatter(x[T, i], d, s=1, c = l[d])

        for _ in range(m):
            for (i, _) in enumerate(distr[T]):
                distr[T, i] = (distr[T, i] + distr[T, i-1])/2

        val = float(np.max(abs(distr[T])))
        plt.scatter(x[T], distr[T], c=distr[T], s=1, cmap='RdYlBu', vmin=-val, vmax=val)
        plt.plot(x[T], np.zeros(x[T].shape), ls='--', c='lightgray')

        plt.ylim(-1.1, 1.1)
        plt.title(f'Spatial distribution of particles at N = {T} for c = {simul.c}')
        plt.xlabel('Position')
        plt.ylabel('State')
        plt.legend()
        if savefig: plt.savefig(f'distr2d_N{T}_c{simul.c}.pdf', bbox_inches='tight')
        plt.show()
    elif proj == '3d':
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection='3d')

        for t in range(T):
            for _ in range(m):
                for (i, _) in enumerate(distr[t]):
                    distr[t, i] = (distr[t, i] + distr[t, i-1])/2
            
            ax.scatter3D([t]*np.size(x[t]), x[t], distr[t], c=distr[t], s=0.5, cmap='RdYlBu')
        
        ax.dist = 11
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_zlabel('State')
        plt.title(f'Spatial distribution of particles between\nN = 0 and N = {T} for c = {simul.c}')
        if savefig: plt.savefig(f'distr3d_N{T}_c{simul.c}.pdf', bbox_inches='tight')
        plt.show()