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

## TODO
- Rad BC?
- Check whether k dot u breaks?
- Check w/ sutherland paper?
