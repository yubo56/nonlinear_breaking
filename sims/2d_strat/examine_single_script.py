#!/usr/bin/env python

import h5py
import matplotlib.pyplot as plt
import numpy as np
dat = h5py.File('snapshots/snapshots_s1/snapshots_s1_p0.h5', mode='r')
tmesh = np.array(dat['tasks']['uz'].dims[0][0])
xmesh = np.array(dat['tasks']['uz'].dims[1][0])
zmesh = np.array(dat['tasks']['uz'].dims[2][0])

uz = dat['tasks']['uz']
bc = uz[:, :, 10]
plt.pcolormesh(tmesh, xmesh, np.transpose(bc))
plt.colorbar()
plt.show()
plt.close()
