import numpy as np
import cv2 as cv 
import math as ma
from matplotlib import pyplot as plt 
import glob
import os
from tkinter import filedialog
import tkinter




#iniating path
path = glob.glob("C:/Users/Acer/OneDrive/Desktop/local/VO/images/*.png")
#appending images in a list
images=[]
for file in path:
	
	image=cv.imread(file,cv.IMREAD_GRAYSCALE)
	images.append(image)
print(len(images))

#caliberation matrix
k =np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02], 
             [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02], 
             [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

#initial rotation and translation matrix and array to append values
translation1=np.zeros((3,1))

rotation1=np.identity(3)
trans_dir=[]
rot_dir=[]
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

#parameters for lkt
lk_params = dict( winSize  = (21,21),maxLevel = 2,criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


#function to extract features
def extract(img):
	fast= cv.FastFeatureDetector_create(threshold = 20, nonmaxSuppression = True)
	kp1 = fast.detect(img)
	kp1 = np.array([kp1[idx].pt for idx in range(len(kp1))], dtype = np.float32)
	print(kp1.shape)
	print(type(kp1))
	return kp1

#funcion for tracking features
def track(img_old,img_new,kp1):

	kp2, st, err = cv.calcOpticalFlowPyrLK(img_old, img_new, np.float32(kp1), None, **lk_params)
	kp2 = kp2[st.ravel()==1]
	kp1 = kp1[st.ravel()==1]
	return kp2,kp1

#function to traingulate feature points to form a point cloud
def triangulate(k,src_p,dst_p,rotation1,translation1,rotation2,translation2):
	a=src_p.shape[0]
	src_p=src_p.reshape((2,a))
	dst_p=dst_p.reshape((2,a))
	proj_mat_i=k@np.hstack((rotation1,translation1))
	proj_mat_f=k@np.hstack((rotation2,translation2))
	cloud=cv.triangulatePoints(proj_mat_i,proj_mat_f,src_p,dst_p)
	return cloud

#funtion for computing the relative scale 
def relative(old_cloud,new_cloud):
	smallest_array = min([new_cloud.shape[0],old_cloud.shape[0]])
	P_X_k = new_cloud[:smallest_array]
	X_k = np.roll(P_X_k,shift = -3)
	P_X_k1 = old_cloud[:smallest_array]
	X_k1 = np.roll(P_X_k1,shift = -3)
	scale_ratio = (np.linalg.norm(P_X_k1 - X_k1,axis = -1))/(np.linalg.norm(P_X_k - X_k,axis = -1))
	return np.median(scale_ratio)




#extracting features from 1st image and tracking it in 2nd
kp1=extract(images[0])
kp2,kp1=track(images[0],images[1],kp1)

#finding essential matrix from the keypoints
F,mask = cv.findFundamentalMat(kp2,kp1,cv.FM_RANSAC ,0.4,0.9,mask=None)
E=k.T@F@k
#checking for rannk
if np.linalg.matrix_rank(E)>2:
	print("no")
#recovering relative translaton and rotation wrt to prev image	
retval,rot,trans,mask=cv.recoverPose(E,kp1,kp2,k)

#updating the rotation and translation ie finding them wrt to 1st image	
rotation2=rot
trans=rotation2@trans
translation2=translation1-trans
trans_dir.append(trans)

rot_dir.append(rotation2)
#triangulating points
new_cloud=triangulate(k,kp1,kp2,rotation1,translation1,rot,trans)
kp3 = kp1

#doing the same as above in a loop
for j in range(1,len(images)-1):
	if j%10==0:
	
	
	lk_params = dict( winSize  = (21,21),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
	
	#tracking features wrt to ref frame keypoints
	kp2,kp3=track(images[j],images[j+1],kp3)
	F,mask = cv.findFundamentalMat(kp2,kp3,cv.FM_RANSAC ,0.4,0.9,mask=None)
	E=k.T@F@k
	if np.linalg.matrix_rank(E)>2:
		print("no")
	retval,rot,trans,mask=cv.recoverPose(E,kp3,kp2,k)
	
	if j==0:
		rotation2=rotation1@rot
	
		trans=rotation2@trans
	
	
		translation2=translation1-trans
		new_cloud=triangulate(k,kp3,kp2,rotation1,translation1,rotation2,translation2)
		translation1=translation2.copy()
		rotation1=rotation2.copy()
		trans_dir.append(translation2)
	
	
		rot_dir.append(rotation2)
	else:
		#finding the rotation and translation wrt to 1st image
		rotation1=rotation2
		translation1=translation2
		old_cloud=new_cloud
		rotation2=rotation1@rot
		trans=rotation2@trans

		#computing cloud of curret and next image by triangulating
		new_cloud=triangulate(k,kp3,kp2,rotation1,translation1,rot,trans)

		#finding relative scale with old cloud and new cloud
		scale=relative(old_cloud,new_cloud)

		#updating translation with scale
		translation2=translation1-scale*trans
		translation1=translation2.copy()
		rotation1=rotation2.copy()
		trans_dir.append(translation2)
	#updating the ref feature points image as they get depleted over certain frames
	if j%20==0:
		kp1=images[j]
	kp3=kp2
		


#calculated trajectory	
figure,axis=plt.subplots()
trans_dir_x=[]
trans_dir_z=[]


for i in range(len(images)-1):
	trans_dir_x.append(trans_dir[i][0][0])
	trans_dir_z.append(trans_dir[i][2][0])

x_truth = []
z_truth = []
# ground truth using pose doc
ground_truth = np.loadtxt('C:\\Users\\Acer\\OneDrive\\Desktop\\local\\VO\\poses.txt')
x_truth=[]
z_truth=[]
for i in range(ground_truth.shape[0]):
    x_truth.append(ground_truth[i,3])
    z_truth.append(ground_truth[i,11])

#plotting both 
plt.plot(x_truth,z_truth, label = "ground_truth")
plt.plot(trans_dir_x,trans_dir_z,color='red',label = "plotted trajectory")
plt.title("monocular camera based plot")
plt.show()