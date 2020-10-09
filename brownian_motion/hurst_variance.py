from fbm import FBM
import numpy as np
import matplotlib.pyplot as plt


def calc_msd_correct(right, left):
    return (right - left) ** 2

iterations = 100

size = 100000
size1 = 20                              # how much particles we have
myarray = np.zeros((size1, size))
myarray2 = np.zeros((size1, size))
myWmsd = np.zeros((size1, size-1))
myWmsd2 = np.zeros((size1, size-1))
coeff = 0.5
f = FBM(size-1, 0.75)
f2 = FBM(size-1, 0.5)

for i in range(myarray.shape[0]):
    myarray[i] = f.fbm()
    myarray2[i] = f2.fbm()


plt.figure()
plt.plot(np.linspace(0, 1, size), myarray[0])
plt.plot(np.linspace(0, 1, size), myarray2[0])
plt.show()


for i in range(myWmsd.shape[0]):
    for j in range(myWmsd.shape[1]):
        myWmsd[i][j] = calc_msd_correct(myarray[i][j], myarray[i][0])
        myWmsd2[i][j] = calc_msd_correct(myarray2[i][j], myarray2[i][0])

itog = np.mean(myWmsd, axis=0)
itog2 = np.mean(myWmsd2, axis=0)
plt.figure()
plt.plot(np.linspace(0, 1, size-1), itog)
plt.plot(np.linspace(0, 1, size-1), itog2)
plt.yscale('log')
plt.xscale('log')
plt.title('MSD(t) for fBM')
plt.show()



# моделирование с разными коэффициентами Херста


myarray_exp = np.zeros((size1, size))
coeff = 0.5
crop = int((size) / 4)

f = FBM(crop, coeff)
print(f.fbm().shape)
coeff += 0.4
f2 = FBM(crop, coeff)
print(f2.fbm().shape)
coeff -= 0.4
f3 = FBM(crop, coeff)
coeff *= 1.3
print(f3.fbm().shape)
f4 = FBM(crop, coeff)
print(f4.fbm().shape)
for i in range(myarray_exp.shape[0]):
    myarray_exp[i][0:crop] = np.array(f.fbm()[:-1])
    lastElement = myarray_exp[i][crop-1]
    myarray_exp[i][crop:crop*2] = np.array(f2.fbm()[1:]) + lastElement

    lastElement = myarray_exp[i][crop*2-1]
    myarray_exp[i][crop*2:crop*3] = np.array(f3.fbm()[1:]) + lastElement
    lastElement = myarray_exp[i][crop*3-1]
    myarray_exp[i][crop*3:size] = np.array(f4.fbm()[1:]) + lastElement
    #print(3)


plt.figure()
plt.plot(np.linspace(0, 1, size), myarray_exp[0])
#plt.plot(np.linspace(0, 1, size), myarray2_exp[0])
plt.show()



# построение MSD
myWmsd_exp = np.zeros((size1, size-1))
for i in range(myWmsd_exp.shape[0]):
    for j in range(myWmsd_exp.shape[1]):
        myWmsd_exp[i][j] = calc_msd_correct(myarray_exp[i][j], myarray_exp[i][0])
        #myWmsd2[i][j] = calc_msd_correct(myarray2[i][j], myarray2[i][0])

itog_exp = np.mean(myWmsd_exp, axis=0)

plt.figure()
plt.plot(np.linspace(0, 1, size-1), itog_exp)
plt.yscale('log')
plt.xscale('log')
plt.title('MSD(t) for fBM')
plt.show()
'''
plt.figure()
plt.plot(t_values, fbm_sample, label = f'h = {f.hurst}')
plt.plot(t_values, fbm_sample_1, label = f'h = {f.hurst}')
plt.plot(t_values, fbm_sample2, label = f'h = {f2.hurst}')
plt.plot(t_values, fbm_sample3, label = f'h = {f3.hurst}')
plt.legend()
plt.title('fractional brownion motion with different hurst exponent')
plt.show()



myLen = t_values.shape[0] - 1
diff_1 = np.zeros(myLen)
diff_1_1 = np.zeros(myLen)

diff_2 = np.zeros(myLen)
diff_3 = np.zeros(myLen)

for i in range(1, t_values.shape[0]):
    diff_1[i - 1] = calc_msd_correct(fbm_sample[0], fbm_sample[i])
    #diff_1_1[i-1] = calc_msd(fbm_sample[0], f)
    diff_2[i - 1] = calc_msd_correct(fbm_sample2[0], fbm_sample2[i])
    diff_3[i - 1] = calc_msd_correct(fbm_sample3[0], fbm_sample3[i])

print(f2.hurst)
plt.figure()
plt.plot(t_values[1:], diff_1, label = f'h = {f.hurst}, alpha = {f.hurst * 2}')
#plt.plot(t_values[1:], diff_1_1, label = f'h = {f.hurst}, alpha = {f.hurst * 2}')
plt.plot(t_values[1:], diff_2, label = f'h = {f2.hurst}, alpha = {f2.hurst * 2}')
plt.plot(t_values[1:], diff_3, label = f'h = {f3.hurst}, alpha = {f3.hurst * 2}')
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.xlim(10**(-3), 10)
plt.ylim(10**(-6), 10**2)
plt.title('Figure of MSD(t) with different hurst exponent')
plt.legend()
plt.show()
'''