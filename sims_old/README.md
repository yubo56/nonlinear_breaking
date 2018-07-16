- `1d_uniform`: 1d uniform background, test waves can propagate
- `2d_no_g`: 2d fluid equations w/ no stratification/no gravitational field
- `2d_strat`: 2d stratified atmosphere, generates videos of single runs
- `2d_strat_conv`: 2d stratified atmosphere, examines convergence as function of
  time/space spacing to analytic solution

- `2d_2_strat_nonlin`: add nonlinear terms, interface forcing
- `2d_2_strat_drop_terms`: try to drop some nonlinear terms to diagnose,
  interface forcing.

- `2d_3_bulk_force`: bulk forcing with a force field instead of BC
- `2d_3_wavepacket`: Sutherland wavepacket
- `2d_3_factored`: try factoring out exp(+- z/2H) from dynamical variables
