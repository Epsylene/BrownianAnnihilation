import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scp

def concentration(simul, plot=None, savefig=False):
        '''
        Return and eventually plot the concentration (as in
        density) of particles in the given simulation over time.

        Args:
            simul: Brownian, Ballistic
                The simulation.
            plot: string
                If 'curve', plot the concentration as a function
                of time. If 'log', plot the concentration as a
                function of time on a log-log scale. Default
                None.
            savefig: bool
                Save the figure in an adequately named PDF file.
        '''
        N = simul.N
        concentration = np.zeros(N)

        # Indices of the particles
        p_idx = np.where(simul.particles == 1)[0]
        x = np.ma.array(simul.x, mask=simul.annihilated)

        # The concentration of particles over time is the number
        # particles divided by the size of the box.
        for t in range(N):
            concentration[t] = np.ma.count(x[t, p_idx])/simul.L

        if plot == 'curve': plt.plot(concentration, c='b')
        elif plot == 'log': plt.loglog(concentration, c='b')

        if plot:
            plt_type = 'logarithmic, ' if plot == 'log' else ''
            plt.title('Density of particles over time\n'
                    + rf'({plt_type}$n_0 = {simul.n}$, $c_0 = {simul.c}$)')
            plt.xlabel('Time')
            plt.ylabel('Concentration')

            if savefig: plt.savefig(f'ctr_n{simul.n}_c{simul.c}_N{simul.N}.pdf', bbox_inches='tight')
            plt.show()

        return concentration

def end_state(simul, params, n_exp):
    '''
    Plots the average final concentrations for a given type of
    simulation over a number of experiments with different
    parameters.

    Args:
        simul: Brownian, Ballistic
            The type of simulation to run.
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

def avg_concentration(simul, params, n_exp, plot=None, savefig=False):
    '''
    Plot an average concentration for a given simulation over a
    number of experiments.

    Args:
        simul: Brownian, Ballistic
            The type of simulation to run.
        params: tuple
            Tuple containing the simulation parameters. The
            tuple (1000, 0.5) would define a Brownian simulation
            with 1000 initial particles and an initial
            concentration of 0.5, for example.
        n_exp: int
            The number of experiments over which to average the
            simulation.
        plot: string
            If 'curve', plot the concentration and average
            concentration curves. If 'log', plot the
            concentration and average concentration on a log-log
            scale. Default None.
        savefig: bool
            Save the figure as an adequately named PDF file.
            Default False.
    '''
    n0, c0, *targs = params
    N = targs[0] if targs else 100

    # Compute concentrations for the provided simulation type
    # and parameters over the number of experiments
    t = np.arange(N)
    c = np.zeros((n_exp, N))
    for i in range(n_exp):
        b = simul(*params)
        c[i] = concentration(b)
    avg_c = np.average(c, axis=0)

    # Plot the concentrations and the average curve
    if plot == 'curve':
        for ci in c: plt.plot(ci, lw=0.5)
        plt.plot(avg_c, c='k')
    elif plot == 'log':
        for ci in c: plt.loglog(ci, lw=0.5)
        plt.loglog(avg_c, c='k')

    if plot:
        plt.title(f'Average concentration over {n_exp}\n'+rf'experiments with $n_0 = {n0}$, $c_0 = {c0}$')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        if savefig: plt.savefig(f'avgc_n0{n0}_c0{c0}_nexp{n_exp}.pdf', bbox_inches='tight')
        plt.show()

    return avg_c, c

def fit(concentration, plot=None, savefig=False):
    '''
    Fit function to find the alpha coefficient of the
    concentration. The fit function is a + b/(t+c)**d, of which
    each parameter can then printed. A R^2 test is also
    performed to measure the goodness of the fit.

    Args:
        concentration: array
            The concentration array to fit, as returned by
            `concentration()` or `avg_concentration()`.
        plot: string
            If 'print', print the fit parameters, R^2 and alpha.
            If 'curve', plot the concentration curve and fit. If
            'log', plot the concentration curve and fit on a
            log-log scale. Default None.
        savefig: bool
            Save the figure in an adequately named PDF file.
            Default False.
    '''
    t = np.arange(len(concentration))
    c_model = lambda t, a, b, c, d: a + b/(t+c)**d

    # Fit the concentration array using the provided function,
    # and calculate the R^2 value, which measures the goodness
    # of the fit. Bounds are provided for the d parameter
    # (alpha) to get a better fit.
    args, _ = scp.curve_fit(c_model, t, concentration, bounds=((-np.inf, -np.inf, -np.inf, 0), (np.inf, np.inf, np.inf, 10)))
    fit = c_model(t, *args)
    r2 = 1 - np.sum((concentration - fit)**2)/np.sum((concentration - np.average(concentration))**2)
    alpha = args[3]

    if plot == 'print':
        # Print the calculated fit parameters and the R^2 test value
        print('Concentration fit a+b/(t+c)^d, with:')
        print('a = {}, b = {},\nc = {}, d = {}'.format(*args))
        print(f'R^2 = {r2}')
    elif plot == 'curve':
        # Plot the concentration and the concentration fit
        plt.scatter(t, concentration, c='k', s=1, label='Simulation')
        plt.plot(t, fit, label='Fit')
    elif plot == 'log':
        plt.loglog(concentration, c='b', label='Simulation')
        plt.loglog(fit, ls='--', c='k', label='Fit')

    if plot:
        plt.title(rf'Concentration fit, $1/t^\alpha$ with $\alpha = {alpha:.2f}$')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()

        if savefig: plt.savefig(f'fit_a{alpha}.pdf', bbox_inches='tight')
        plt.show()

    return fit, alpha

def distribution(simul, m=None, T=None, plot='2d', savefig=False):
    '''
    Compute the average spatial distribution of particles over
    time as a position-state plot. The "average state" at each
    point is a number between -1 and 1, computed as the
    neighbor-by-neighbor average at that point between "particle
    state" (1), "antiparticle state" (-1) and "vacuum state"
    (0).

    Args:
        simul: Brownian, Ballistic object
            A pre-computed simulation.
        m: int
            The number of recursion steps for the averaging
            function. The bigger it is, the smoother the
            position-state curve will be, but also closer to 0.
            Default 50 for 2D projection and 10 for 3D
            projection.
        T: int
            The simulation step at which (if 2D projection is
            chosen) or up to which (3D projection) the
            distribution is calculated. Default the last step of
            the simulation.
        plot: '2d', '3d'
            If 2D, plot the average state of each point of the
            box at the time step T. The actual particles and
            antiparticles are also plotted on top and bottom, to
            better visualise the average state curve. If 3D,
            plot the average state of each point of the box
            between 0 and T. Default 2D.
        savefig: bool
            If True, save the plot figure in an adequately named
            PDF file. Note that this can be quite slow for the
            3D plot.
    '''
    n, N, L = simul.n, simul.N, simul.L
    if T == None: T = simul.N-2
    if m == None: m = 50 if plot == '2d' else 10

    x = simul.space
    distr = np.zeros(x.shape)
    part = simul.particles

    # Position the particle and anti-particle states on the box
    # space array
    for t in range(T+1):
        for (i, xi) in enumerate(simul.x[t]):
            # Because the box space and the particles positions
            # arrays are of different sizes, we have to check
            # which points are the same in each...
            idx = np.where(abs(x[t] - xi) < 0.001)
            # ...and at those indices write the particle and
            # anti-particle states (everywhere else is the vacuum,
            # which is 0 state).
            distr[t, idx] = part[i]

    if plot == '2d':
        # In order to write the labels only once
        plt.scatter(np.nan, np.nan, c='b', s=1, label='Particles')
        plt.scatter(np.nan, np.nan, c='r', s=1, label='Anti-particles')
        
        # Plot the actual particles and anti-particles, in blue
        # and red, correspondingly
        for (i, d) in enumerate(distr[T]):
            l = {1: 'b', -1: 'r'}
            if d != 0:
                plt.scatter(x[T, i], d, s=1, c = l[d])

        # Averaging the distribution: take a point, average its
        # state with its left neighbor, and repeat the operation
        # for each point `m` times.
        for _ in range(m):
            for (i, _) in enumerate(distr[T]):
                distr[T, i] = (distr[T, i] + distr[T, i-1])/2

        # Plot the distribution as a state-position scatter
        # curve, with a red-blue gradient colormap ranging from
        # the distribution's greatest point (in absolute value)
        # to its opposite (so that values above 0 are mostly
        # blue --particle-like-- and values below 0 are mostly
        # red --anti-particle-like--).
        distr_range = float(np.max(abs(distr[T])))
        plt.scatter(x[T], distr[T], c=distr[T], s=1, cmap='RdYlBu', vmin=-distr_range, vmax=distr_range)
        # Plot a dashed gray line at 0, as a reference point.
        plt.plot(x[T], np.zeros(x[T].shape), ls='--', c='lightgray')

        plt.ylim(-1.1, 1.1)
        plt.title(f'Spatial distribution of particles at N = {T} for c = {simul.c}')
        plt.xlabel('Position')
        plt.ylabel('State')
        plt.legend()
        if savefig: plt.savefig(f'distr2d_N{T}_c{simul.c}.pdf', bbox_inches='tight')
        plt.show()
    elif plot == '3d':
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection='3d')

        # Average the distribution over time in the [0, T[
        # range.
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
        if savefig: plt.savefig(f'distr3d_N{T}_c{simul.c}.png', bbox_inches='tight')
        plt.show()