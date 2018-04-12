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
