from fbm import FBM
import numpy as np
import matplotlib.pyplot as plt


def calc_msd(sample_diff):
    return np.sum((np.diff(sample_diff)) ** 2)


f = FBM(5000, 0.75)
f2 = FBM(5000, 0.999)
f3 = FBM(5000, 0.3)
# Generate a fBm realization
fbm_sample = f.fbm()
fbm_sample2 = f2.fbm()
fbm_sample3 = f3.fbm()
# Get the times associated with the fBm
t_values = f.times()


plt.figure()
plt.plot(t_values, fbm_sample, label = 'h = 0.75')
plt.plot(t_values, fbm_sample2, label = 'h = 0.999')
plt.plot(t_values, fbm_sample3, label = 'h = 0.3')
plt.legend()
plt.title('fractional brownion motion with different hurst exponent')
plt.show()



myLen = t_values.shape[0] - 1
diff_1 = np.zeros(myLen)
diff_2 = np.zeros(myLen)
diff_3 = np.zeros(myLen)

for i in range(1, t_values.shape[0]):
    diff_1[i - 1] = calc_msd(fbm_sample[:i])
    diff_2[i - 1] = calc_msd(fbm_sample2[:i])
    diff_3[i - 1] = calc_msd(fbm_sample3[:i])

plt.figure()
plt.plot(t_values[1:], diff_1, label = 'h = 0.75')
plt.plot(t_values[1:], diff_2, label = 'h = 0.999')
plt.plot(t_values[1:], diff_3, label = 'h = 0.3')
plt.yscale('log')
plt.xscale('log')
plt.title('Figure of MSD(t) with different hurst exponent')
plt.legend()
plt.show()
