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

## TODO
- plot spatial FT as function of time

# GLOBAL TODO
- Rad BC?
- Analytically follow Sutherland modulationally stable/unstable proof
- Check w/ sutherland paper?
