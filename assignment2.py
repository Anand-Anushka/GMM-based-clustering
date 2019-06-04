import cv2
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


img = cv2.imread('ski_image.jpg')
img=numpy.divide(img,10.0) #for normalization
size=(img.shape[0]*img.shape[1])
d=3
cov=numpy.identity(d)
N=[[],[],[]] #Nk
u=[[],[],[]] #means

'Initialization of means'
u[0]=(12,12,12)
u[1]=(1.2,1.2,1.2)
u[2]=(18,18,18)

cov=[[],[],[]] #covarience matrices
cov[0]=numpy.identity(d)
cov[1]=numpy.identity(d)
cov[2]=numpy.identity(d)

pi=(1./3,1./3,1./3) #priors

v=[[],[],[]] #responsibility matrices
v[0]=numpy.zeros((img.shape[0],img.shape[1]))
v[1]=numpy.zeros((img.shape[0],img.shape[1]))
v[2]=numpy.zeros((img.shape[0],img.shape[1]))

s1=[[],[],[]] #for pi[i]*N(u,cov)
likelihood=[]
covtemp=[[],[],[]]
covtemp[0]=numpy.zeros((3,3,size))
covtemp[1]=numpy.zeros((3,3,size))
covtemp[2]=numpy.zeros((3,3,size))
for ctr in range (50):
    '50 iterations to show convergence of log likelihood function' 
    'E step'
    
    N[0]=pi[0]*multivariate_normal.pdf(img, mean=u[0], cov=cov[0])
    N[1]=pi[1]*multivariate_normal.pdf(img, mean=u[1], cov=cov[1])
    N[2]=pi[2]*multivariate_normal.pdf(img, mean=u[2], cov=cov[2])
    s=N[0]+N[1]+N[2]
    v=N/s #responsibility calculated
    N1=[[sum(sum(v[0]))],[sum(sum(v[1]))],[sum(sum(v[2]))]]
    'M step'
    #pi calculated
    pi=numpy.divide(numpy.sum(numpy.sum(v,2),1),img.shape[0]*img.shape[1])
    print(pi)
    img1=numpy.rollaxis(img,2)
    
    t1=numpy.sum(numpy.sum(numpy.multiply(img1,v[0]),2),1)
    t2=numpy.sum(numpy.sum(numpy.multiply(img1,v[1]),2),1)
    t3=numpy.sum(numpy.sum(numpy.multiply(img1,v[2]),2),1)
    u[0]=numpy.divide(t1,N1[0])
    u[1]=numpy.divide(t2,N1[1])
    u[2]=numpy.divide(t3,N1[2])#means calculated
    'For calculation of covariance matrices'
    sub1=img-u[0]
    sub1=numpy.reshape(sub1,(size,3))
    sub2=img-u[1]
    sub2=numpy.reshape(sub2,(size,3))
    sub3=img-u[2]
    sub3=numpy.reshape(sub3,(size,3))
    v1=numpy.reshape(v,(3,size))
    
    for i in range(size):
        covtemp[0][:,:,i]=numpy.dot(sub1[i][:,None],sub1[i][None])
        covtemp[1][:,:,i]=numpy.dot(sub2[i][:,None],sub2[i][None])
        covtemp[2][:,:,i]=numpy.dot(sub3[i][:,None],sub3[i][None])
        
    m=numpy.multiply(covtemp[0],v1[0])
    cov[0]=numpy.divide(numpy.sum(m,axis=2),N1[0])+0.1*numpy.identity(d)
    
    m=numpy.multiply(covtemp[1],v1[1])
    cov[1]=numpy.divide(numpy.sum(m,axis=2),N1[1])+0.1*numpy.identity(d)
    
    m=numpy.multiply(covtemp[2],v1[2])
    cov[2]=numpy.divide(numpy.sum(m,axis=2),N1[2])+0.1*numpy.identity(d)
    'For computation of log likelihood'
    s1[0]=pi[0]*multivariate_normal.pdf(img, mean=u[0], cov=cov[0])
    s1[1]=pi[1]*multivariate_normal.pdf(img, mean=u[1], cov=cov[1])
    s1[2]=pi[2]*multivariate_normal.pdf(img, mean=u[2], cov=cov[2])
    likelihood.append(sum(sum(numpy.divide(numpy.log(sum(s1)),100000))))


vfinal=numpy.argmax(v,0) #gaussian with max responsibility is choosen
'The segmented image is assigned to the mean of the choosen gaussian'
img_seg=numpy.zeros(img.shape) 
img_seg_rgb=numpy.zeros(img.shape) 
color=[[20,0,0],[0,20,0],[0,0,20]]

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_seg[i][j]=u[vfinal[i][j]]
        img_seg_rgb[i][j]=color[vfinal[i][j]]
        
'To display segmented image'    
plt.plot(likelihood)  
plt.ylabel('Log likelihood')
plt.xlabel('Iterations')  
plt.title('Log likelihood plot showing saturation') 
plt.grid(True)
plt.show()
t=numpy.multiply(img_seg,10 )
cv2.imwrite('segmented_img.jpg',t.astype(numpy.uint8))
#cv2.imshow('ImageWindow',t.astype(numpy.uint8))
#cv2.waitKey()     

t=numpy.multiply(img_seg_rgb,10 )
cv2.imwrite('segmented_img_rgb.jpg',t.astype(numpy.uint8))
#cv2.imshow('ImageWindow',t.astype(numpy.uint8))
#cv2.waitKey()     