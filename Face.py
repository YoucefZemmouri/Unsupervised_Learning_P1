from tools import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# plt.ion()  # enable interactive plotting

def show_images(ims):
    im_large = np.zeros((shape[0],shape[1]*ims.shape[0]))
    for i in range(10):
        img = ims[i, :].reshape(shape)
        im_large[:,shape[1]*i:shape[1]*(i+1)] = img
    plt.imshow(im_large,cmap='gray')
    plt.show()

imgs = []
shape = (192,168)
for i in range(10):
    img = mpimg.imread('images/1_%02d.pgm' % (i+1))
    v = img.flatten()
    imgs.append(v)

imgs = np.array(imgs)

# Choose observed entries at random
gamma = 0.1  # ratio of missing entries
indices = np.random.choice(imgs.size, int(gamma * imgs.size), replace=False)
W = np.ones(imgs.shape)
for i in indices:
    W.itemset(i,0)  # flatten indexing

imgs_miss = W*imgs


tau = 50000
beta = 0.01
epsilon = 0.01

imgs_recover = LRMC(imgs_miss, W, tau, beta, epsilon)

plt.figure(0)
show_images(imgs_miss)
show_images(imgs_recover)

plt.show()