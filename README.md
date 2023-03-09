# Brownian/ballistic annihilation

> Diffusion-reaction processes can model many interesting situations. They are used for instance to model disease
propagation in the same way as chemical reactions: H+S → 2S (contamination), S → H (healing), S→ ∅ (death)
with H healthy and S sick. Depending on the reactive particles dynamical properties (ballistic or brownian), on
the space dimensionality, etc, interesting phenomena are observed.
In 1983 D. Toussaint and F. Wilczek considered a simple situation of 2 reactive Brownian species A, A* obeying
A + A* → ∅. This was meant to be a toy description of the matter-antimatter annihilation that might have occurred
in the early cosmological times. They were interested in finding out whether such model could predict a local
excess of matter over antimatter. It turned out that this model provides a simple and non trivial example of case
where usual chemical kinetics fails to predict the observed situation in low dimensional spaces.

We use here two approaches to model diffusion-reaction processes in 1D, where mean field chemical kinetics fail, in the simplified case of a mixture of two species A and A* obeying the reaction A + A* → ∅:

1. **Ballistic annihilation**: particles start with random positions and velocities along an axis, and are simulated assuming elastic collisions;
2. **Brownian annihilation**: particles move with constant randomly negative or positive increments on a lattice (Brownian walk).