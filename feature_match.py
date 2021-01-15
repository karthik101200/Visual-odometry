import numpy as np
import cv2 as cv 
import math as ma
from matplotlib import pyplot as plt 
import glob
import os
from tkinter import filedialog
import tkinter

#initiating the path
path = glob.glob("C:/Users/Acer/OneDrive/Desktop/local/VO/images/*.png")
#creating a list to store images
images=[]
for file in path:
	
	image=cv.imread(file,cv.IMREAD_GRAYSCALE)
	images.append(image)
print(len(images))
#camera caliberation matrix
k =np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02], 
             [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02], 
             [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
#initial translation and rotation
translation1=np.zeros((3,1))

rotation1=np.identity(3)
trans_dir=[]
rot_dir=[]
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
#function for extracting features
def extract(img1,img2):
	# Initiate SIFT detector
	sift = cv.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# BFMatcher with default params
	bf = cv.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

		# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	src_p = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_p=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

	return src_p,dst_p
#function for triangulating feature points and makng a point cloud
def triangulate(k,src_p,dst_p,rotation1,translation1,rotation2,translation2):
	a=src_p.shape[0]
	

	src_p=src_p.reshape((2,a))
	
	dst_p=dst_p.reshape((2,a))
	
	proj_mat_i=k@np.hstack((rotation1,translation1))
	proj_mat_f=k@np.hstack((rotation2,translation2))
	cloud=cv.triangulatePoints(proj_mat_i,proj_mat_f,src_p,dst_p)
	return cloud
#function for computing scale with calculated point clouds
def relative(old_cloud,new_cloud):
	smallest_array = min([new_cloud.shape[0],old_cloud.shape[0]])
	P_X_k = new_cloud[:smallest_array]
	X_k = np.roll(P_X_k,shift = -3)
	P_X_k1 = old_cloud[:smallest_array]
	X_k1 = np.roll(P_X_k1,shift = -3)
	scale_ratio = (np.linalg.norm(P_X_k1 - X_k1,axis = -1))/(np.linalg.norm(P_X_k - X_k,axis = -1))
	return np.median(scale_ratio)

for j in range(len(images)-1):

	src_p,dst_p=extract(images[j],images[j+1])
	#calculating essential matrix from key points	
	F,mask = cv.findFundamentalMat(src_p,dst_p,cv.FM_RANSAC ,0.4,0.9,mask=None)
	E=k.T@F@k
	#check for rank 3 matrix
	if np.linalg.matrix_rank(E)>2:
		print("no")
	#recovering rot and translating wrt to prev frame
	retval,rot,trans,mask=cv.recoverPose(E,src_p,dst_p,k)
	#updating rotation and translation wrt to 1st frame
	if j==0:
		
		rotation2=rotation1@rot
	
		trans=rotation2@trans
		translation2=translation1-trans
		#making point cloud by triangulating
		new_cloud=triangulate(k,src_p,dst_p,rotation1,translation1,rotation2,translation2)
		translation1=translation2.copy()
		rotation1=rotation2.copy()
		trans_dir.append(translation2)
	
	
		rot_dir.append(rotation2)
	else:
		rotation1=rotation2
		translation1=translation2
		old_cloud=new_cloud
		rotation2=rotation1@rot
	
		trans=rotation2@trans
	
	
		#making point cloud by triangulating
		new_cloud=triangulate(k,src_p,dst_p,rotation1,translation1,rot,trans)
		#computing scale
		scale=relative(old_cloud,new_cloud)
		
		translation2=translation1-scale*trans
		translation1=translation2.copy()
		rotation1=rotation2.copy()
		trans_dir.append(translation2)
		


#for computed trajectory
figure,axis=plt.subplots()
trans_dir_x=[]
trans_dir_z=[]


for i in range(0,149):
	trans_dir_x.append(-trans_dir[i][0][0])
	trans_dir_z.append(trans_dir[i][2][0])

#for ground truth
x_truth = []
z_truth = []
# ground truth using pose doc
ground_truth = np.loadtxt('C:\\Users\\Acer\\OneDrive\\Desktop\\local\\VO\\poses.txt')
x_truth=[]
z_truth=[]
for i in range(ground_truth.shape[0]):
    x_truth.append(ground_truth[i,3])
    z_truth.append(ground_truth[i,11]) 

#plotting graphs
plt.plot(x_truth,z_truth, label = "ground_truth")
plt.plot(trans_dir_x,trans_dir_z,color='red',label = "plotted trajectory")
plt.title("monocular camera based plot")
plt.show()