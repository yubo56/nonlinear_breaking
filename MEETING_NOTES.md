# TODO future work

massive stars, main theory zahn etc

for circ orbit, Kushnir/Zaldarriaga has analytical solution (dynamical tides
simplified) for torque

What about high e? decompose into spherical modes (n * P), can just add up
torques for a bunch of circular modes. But is tedious. Can get closed form like
weak tides?

Vick & Lai, eccentric path, (S4) gives energy/torques for each component.

But for massive stars, torques might be analytic (K&Z).

Strongest might be pericenter, would have just one term. How much error? might
be enough?

Also maybe Storch \& Lai viscoelastic?

# NOTES 04/17/18
- Creating nonlinear equations: need pressure term on LHS, need a full-rank
  linear operator for basis by which to expand nonlinear terms

- Little wiggles = IC + BC weren't div-free!
    - show `ns_sponge_lin` at early times

    - Tried to smoothly introduce, still wiggles, need to be identically ok for
      strongly nonlin
    - show `ns_sponge_nonlin`, `ns_sponge_nonlin_gradual`

    - Remove dz(ux) (disp term) to free ux (no sharp spikes)
    - Use neumann BC instead? Seems smoother, still unstable for large A

- Seems incompressible + driving BC + zero IC is pathological!
    - MHD simulations (Daniel + last week paper) use driving force instead of BC
        - Not great for our problem! We have a wave propagating, not driving
    - Do not streamfunction, 2D only
    - Anelastic? Could work out nonlinear terms, no div=0 constraints

# NOTES 04/24/18
- rho1 << rho0 = incompressibility = v << c\_s, go ahead and take linear approx!
    - think my janky prescription was introducing floating point error
    - may need to consider the fully compressible case for astrophysical systems
      of interest...
- consider moving to anelastic equations? see how it compares to incompressible

# NOTES 05/01/18
- still divergences when rho1 << rho!
- Think Sutherland argument "lin sol is exact sol of nonlinear" is incorrect
- Consider truly reduced problem, 1d wave equation
    - zero ic, driving + rad BC, weak nonlinear term

# NOTES 06/04/18
- No breaking with bulk driving, small artifacts probably related to bad
  damping?
- Try wavepacket, high k divergences
- high k instabilities: due to improperly resolving the Gaussian, bad BC
    - delta P is from nonlinear term having nonzero integral

# NOTES 07/13/18 Daniel call
- Linearly increasing velocity at excitation zone? Probably can ignore
- Damp nonlinear terms near the excitation zone
- Explicit = RHS, use damping zone on RHS.
    Implicit will give much higher-order Chebyshev coefficients on LHS, much
    less sparse matricies
- Store LU decomp!!
- Try to factor out all exponentials?
- Sutherland finite diff vertical?
    - Chebyshev is more accurate especially for wave-wave.
- Gibbs ringing when lose exponential convergence
- Probably use sigma << 1/kz

# NOTES 07/20/18 Daniel call
- Add Navier-Stokes viscosity
- Use larger Lx so can use larger kx, physical
- Try to increase/decrease kz to reproduce Sutherland paper

- Try to analytically predict when code breaks down? Floquet theory
- Ultimate goal is to get the width and distribution of "breaking layer"

# NOTES 08/06
- Mask nonlinear terms in no-NS? Need to be able to run low kx models
    - Alt: gradually turn on driving force
- Triple check all timescales in NS... wtf
    - Consider correct damping timescales?
- Consider artificial regularization

# NOTES 09/17
- Solve linear corotation resonance for cylindrical/spherical geometry (Dong
  thinks spherical might be too hard)
    - Possibly think about when nonlinear effects kick in?
- Use smaller kz, such that nu/dz^2 << w << nu kz^2
- Plot uz(kx, z, t), see what height does uz spread out and dissipate.
- Eventual goal, recall, is dP/dz, d^2(px)/dzdt, gives timescale of formation of
  critical layer, upon which our previous work becomes usable.

# NOTES 09/31
- Meeting plan:
    - NSF
    - `2d_1_gradual`: Show off correct regularization, had to mask NS plus
      increased width driving zone
    - Hopefully show off critical layer transmission over different Ri?
    - Analytical work:
        - Differ from David's b/c assumed |mu - 0.5| >> 1. Verify our two
          expressions agree otherwise
        - Repeated David's analysis for plane parallel, also have a log term!
        - Ask what Dong thinks about dispersion relation shortcut?
- Try plane parallel for finite sound speed?

# NOTES 02/15
- try NL mask in forcing zone? see impact on A(t) or at least understand forcing
  in general
- Once we figure this out, try higher res!
- Compare reflection/transmission to B&B67, Winters & asaro 92

- story should be fine for astro
- If wanted to try JFM, really pin down numerics (forcing) R(nu) won't need to
  be as well pinned dow though.
- JFM rapids? ~ ApJL

# NOTES (upcoming)
- checked my integrations, integrating against (kx, kz) + 0.5 * (kx, -kz) + 0.5
  * (2kx, -2kz) recovers correct amplitudes (to 0.0003)
- TODO: convert all indicies to linear interpolations?

# GLOBAL TODO
- E disp
- Finish trying power law calculation?
