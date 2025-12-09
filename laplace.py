#solving the first elliptic equation by jacobi method.
#this method involve just iteration
#bounadary potential fixed to be 1volt

import numpy as np
import matplotlib.pyplot as plt

m=100 # no of squares that is grid will be 101*101

v=1.0

targ=1e-6 #acuuracy

phi=np.zeros([m+1,m+1]) #grid specified

phi[0,:]=v  # remember in pytthon indices started from 0 not 1 so 100 which is m means its already 101th point
phi[m,:]=v
phi[:,0]=v
phi[:,m]=v

phiprime=np.zeros_like(phi)

delta=1.0

while delta>targ:

    for i in range(m+1):
        for j in range(m+1):
            if i==0 or i==m or j==0 or j==m:
                phiprime[i,j]=phi[i,j]
            else:
                phiprime[i,j]=(phi[i+1,j]+phi[i-1,j]+phi[i,j+1]+phi[i,j-1])/4.0

    delta=np.max(np.abs(phi-phiprime))

    phi,phiprime=phiprime,phi

plt.imshow(phi,origin='lower')
plt.colorbar()
plt.show()
