import cv2
import numpy as np
import math
import statistics



img=cv2.imread('p003.png') #path to the image
img=np.float64(img)
blue,green,red=cv2.split(img)

blue[blue==0]=1
green[green==0]=1
red[red==0]=1

div=np.multiply(np.multiply(blue,green),red)**(1.0/3)

a=np.log1p((blue/div)-1)
b=np.log1p((green/div)-1)
c=np.log1p((red/div)-1)

a1 = np.atleast_3d(a)
b1 = np.atleast_3d(b)
c1 = np.atleast_3d(c)
rho= np.concatenate((c1,b1,a1),axis=2) #log chromaticity on a plane

U=[[1/math.sqrt(2),-1/math.sqrt(2),0],[1/math.sqrt(6),1/math.sqrt(6),-2/math.sqrt(6)]]
U=np.array(U) #eigens

X=np.dot(rho,U.T) #2D points on a plane orthogonal to [1,1,1]


d1,d2,d3=img.shape

e_t=np.zeros((2,181))
for j in range(181):
    e_t[0][j]=math.cos(j*math.pi/180.0)
    e_t[1][j]=math.sin(j*math.pi/180.0)

Y=np.dot(X,e_t)
nel=img.shape[0]*img.shape[1]

bw=np.zeros((1,181))

for i in range(181):
    bw[0][i]=(3.5*np.std(Y[:,:,i]))*((nel)**(-1.0/3))

entropy=[]
for i in range(181):
    temp=[]
    comp1=np.mean(Y[:,:,i])-3*np.std(Y[:,:,i])
    comp2=np.mean(Y[:,:,i])+3*np.std(Y[:,:,i])
    for j in range(Y.shape[0]):
        for k in range(Y.shape[1]):
            if Y[j][k][i]>comp1 and Y[j][k][i]<comp2:
                temp.append(Y[j][k][i])
    nbins=round((max(temp)-min(temp))/bw[0][i])
    (hist,waste)=np.histogram(temp,bins=nbins)
    hist=filter(lambda var1: var1 != 0, hist)
    hist1=np.array([float(var) for var in hist])
    hist1=hist1/sum(hist1)
    entropy.append(-1*sum(np.multiply(hist1,np.log2(hist1))))

angle=entropy.index(min(entropy))

e_t=np.array([math.cos(angle*math.pi/180.0),math.sin(angle*math.pi/180.0)])
e=np.array([-1*math.sin(angle*math.pi/180.0),math.cos(angle*math.pi/180.0)])

I1D=np.exp(np.dot(X,e_t)) #mat2gray to be done


p_th=np.dot(e_t.T,e_t)
X_th=X*p_th
mX=np.dot(X,e.T)
mX_th=np.dot(X_th,e.T)

mX=np.atleast_3d(mX)
mX_th=np.atleast_3d(mX_th)

theta=(math.pi*float(angle))/180.0
theta=np.array([[math.cos(theta),math.sin(theta)],[-1*math.sin(theta),math.cos(theta)]])
alpha=theta[0,:]
alpha=np.atleast_2d(alpha)
beta=theta[1,:]
beta=np.atleast_2d(beta)




#Finding the top 1% of mX
mX1=mX.reshape(mX.shape[0]*mX.shape[1])
mX1sort=np.argsort(mX1)[::-1]
mX1sort=mX1sort+1
mX1sort1=np.remainder(mX1sort,mX.shape[1])
mX1sort1=mX1sort1-1
mX1sort2=np.divide(mX1sort,mX.shape[1])
mX_index=[[x,y,0] for x,y in zip(list(mX1sort2),list(mX1sort1))]
mX_top=[mX[x[0],x[1],x[2]] for x in mX_index[:int(0.01*mX.shape[0]*mX.shape[1])]]
mX_th_top=[mX_th[x[0],x[1],x[2]] for x in mX_index[:int(0.01*mX_th.shape[0]*mX_th.shape[1])]]
X_E=(statistics.median(mX_top)-statistics.median(mX_th_top))*beta.T
X_E=X_E.T

for i in range(X_th.shape[0]):
   for j in range(X_th.shape[1]):
       X_th[i,j,:]=X_th[i,j,:]+X_E

rho_ti=np.dot(X_th,U)
c_ti=np.exp(rho_ti)
sum_ti=np.sum(c_ti,axis=2)
sum_ti=sum_ti.reshape(c_ti.shape[0],c_ti.shape[1],1)
r_ti=c_ti/sum_ti

r_ti2=255*r_ti


cv2.imwrite('p003-1.png',r_ti2) #path to directory where image is saved
