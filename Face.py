from tools import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# plt.ion()  # enable interactive plotting

def hstack_images(ims):
    im_large = np.zeros((shape[0],shape[1]*ims.shape[0]))
    for i in range(10):
        img = ims[i, :].reshape(shape)
        im_large[:,shape[1]*i:shape[1]*(i+1)] = img
    return im_large

imgs = []
shape = (192,168)
individual = 1
for i in range(10):
    img = mpimg.imread('images/%d_%02d.pgm' % (individual, i+1))
    v = img.flatten()
    imgs.append(v)

imgs = np.array(imgs)

# Choose observed entries at random
gamma = 0.3  # ratio of missing entries
indices = np.random.choice(imgs.size, int(gamma * imgs.size), replace=False)
W = np.ones(imgs.shape)
for i in indices:
    W.itemset(i,0)  # flatten indexing
M = np.count_nonzero(W)  # expected number of observed entries


imgs_miss = W*imgs



tau = 20000
# Leman: beta should be strictly < 2, beta = 2 may diverge, no idea why ...
beta = min(1.9, imgs.size/M)
epsilon = 0.01  # for reference of unit, pixel value range in [0, 255]
print('(tau,beta,eps)=', tau, beta, epsilon)

imgs_recover = LRMC(imgs_miss, W, tau, beta, epsilon)

MSE = np.mean((imgs_recover - imgs) ** 2)
print('Mean Square Error = ', MSE)

row0 = hstack_images(imgs)
row1 = hstack_images(imgs_miss)
row2 = hstack_images(imgs_recover)
row012 = np.vstack((row0, row1, row2))
plt.imshow(row012, cmap='gray')
plt.show()

for i in range(10):
    name = 'result/missing_%03d_%d_%02d.png' % (gamma*100, individual, i+1)
    mpimg.imsave(name, imgs_miss[i].reshape(shape), vmin=0, vmax=255, cmap='gray')

for i in range(10):
    name = 'result/recover_%03d_tau_%d_%d_%02d_.png' % (gamma * 100, tau, individual, i + 1)
    mpimg.imsave(name, imgs_recover[i].reshape(shape), vmin=0, vmax=255, cmap='gray')

# epsilon = 0.01

# 90% missing
# tau = [1000, 20000, 400000, 8000000]
# MSE = [12078, 10739, 6860, 6556*]
# * epsilon = 0.1

# 70% missing
# tau = [1000,20000, 400000, 8000000]
# MSE = [9213,5853, 1753, 1602*]
# * epsilon = 0.1

# 50% missing
# tau = [1000,20000, 400000, 8000000]
# MSE = [6439,2592, 467, 420]

# 30% missing
# tau = [1000, 20000, 400000, 8000000]
# MSE = [3728, 713, 121, 116]