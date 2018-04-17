# nonlinear wave breaking research project
- Uses [Dedalus](https://bitbucket.org/dedalus-project/dedalus), can just clone
  when setting up: `hg clone ssh://hg@bitbucket.org/dedalus-project/dedalus`

# Anaconda install notes (need to relog after updating virtualenv?):
```
conda install virtualenv
conda install -c cryoem openmpi=2.0.2-0
conda install -c conda-forge pyfftw

mkdir -p ~/fftw
cd ~/fftw
wget http://www.fftw.org/fftw-3.3.7.tar.gz
tar -xzf fftw-3.3.7.tar.gz
mv fftw-3.3.7/* . && rmdir fftw-3.3.7 && rm fftw-3.3.7.tar.gz

./configure --enable-mpi --enable-openmp --enable-shared --with-pic --disable-fortran --prefix=/data2/yubosu/fftw
make
make install
```

Next, need to go into `dedalus/setup.py` and fix the commented out section. Finally, can do
```
cd ~/research/nonlinear_breaking/dedalus/
MPI_PATH=~/anaconda3 FFTW_PATH=~/fftw make
```

# NOTES
- Pressure term: need on LHS, need a full-rank linear operator for basis by
  which to expand nonlinear terms
- Little wiggles = IC + BC weren't div-free!
    - Tried to smoothly introduce, still wiggles, need to be identically ok for
      strongly nonlin
    - Remove dz(ux) (disp term) to free ux (no sharp spikes)
    - Use neumann BC instead? Seems smoother

- Seems incompressible + driving BC + zero ICis pathological!
    - MHD simulations (Daniel + last week paper) use driving force instead of BC
        - Not great for our problem! We have a wave propagating, not driving
    - Do not streamfunction, 2D only
    - Anelastic? Could work out nonlinear terms, no div=0 constraints

- To generate:
    - plots showing dz(ux) at small t spike
    - video for gradual driving increase at small and large A
- Rad BC? TODO
- How handle non-evolution constraint in matrix elements?
