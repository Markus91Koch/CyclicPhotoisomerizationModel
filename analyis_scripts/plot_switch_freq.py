import numpy as np
import matplotlib.pyplot as plt
import os

tc = np.loadtxt("switching_freq/tc_count.txt")
ct = np.loadtxt("switching_freq/ct_count.txt")


trans = np.loadtxt("switching_freq/trans_count.txt")
cis = np.loadtxt("switching_freq/cis_count.txt")



fig, (ax1,ax2) = plt.subplots(2,sharex=True)


ax1.plot(np.arange(len(tc)), tc, "r-", label="trans-cis")
ax1.plot(np.arange(len(ct)), ct, "b-", label="cis-trans")
ax1.legend(loc="best")

ax2.plot(np.arange(len(trans)), trans, "r-", label="trans")
ax2.plot(np.arange(len(cis)), cis, "b-", label="cis")
ax2.legend(loc="best")

plt.show()
