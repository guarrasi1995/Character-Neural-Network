import numpy as np

michele = np.load("data_michele.npy")



import matplotlib.pyplot as plt
modifica = np.resize(michele[0], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[1], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[2], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[3], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[4], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[5], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[6], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[7], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[8], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[9], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[10], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

modifica = np.resize(michele[-2], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

np.max(michele[4])

for i in range(len(michele)):
    michele[i] = (michele[i] -np.min(michele[i]))/np.max(michele[i])













0.004 = 1/250
max(michele[9])












m= np.load("../data/dirty_data.npy")
modifica = np.resize(m[6], (64, 64))
imgplot = plt.imshow(modifica)
plt.colorbar(imgplot)

