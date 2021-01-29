## MONOCULAR VISUAL ODOMETRY


##### Introduction:

Monocular visual odometry is aimed to get the estimated motion of a calibrated camera which is mounted on a vehicle. Hence it is possible to obtain the trajectory of the vehicle from the images taken from the camera.
This camera(s) system coupled with other sensors like lidar and imu can give an optimal trajectory which is extremely useful in localisation and motion planning.Here we take images shot from a single camera, process them to obtain a trajectory and we compare it with a ground truth.

The algorithm was coded in python(3) with OpenCV and other libraries.

The algorithm was implemented on dataset of 150 images provided by KITTI. The results were also confirmed by applying it on other datasets.   

#### Method (Feature Tracking):

1)Features(key points) in the first frame(reference) were identified by FAST.

2)These features were tracked in the 2nd frame(next frame) using the LKT optical flow algorithm(Lucas kanade tracking).

3)Essential matrix was obtained by given features and camera parameters and then decomposed to obtain relative translation and rotation.

4)With the relative rotation and translation point cloud(3D) of the features were obtained wrt to 1st frame by triangulation the 2D key points.

5)Now the 2nd frame was taken as the reference all previous steps (1-4) were repeated to get features in the next frame and a cloud of features of the 2nd and 3rd frame was obtained.

6)Relative scale between frames was obtained by taking the norm (mean distance ) between two clouds.

7)Now we update the rotation and translation wrt 1st frame.

8)Above steps are repeated for the whole dataset of images and we plot the translation(trajectory)wrt to 1st frame.

#### NOTE-Relative translation(translation wrt 1st frame) was calculated as we require global pose / ground truth for absolute translation.

#### Method(Feature Matching):
1)Features in the 1st and 2nd frame were identified and matched using SIFT and Brute force matching(BF) along with KNN.

2)Essential matrix was obtained by given features and camera parameters and then decomposed to obtain relative translation and rotation.

3)With the relative rotation and translation point cloud(3D) of the features were obtained wrt to 1st frame by triangulation the 2D key points.

4)Now the 2nd frame was taken as the reference and all previous steps (1-3) were repeated to get features in the next frame and a cloud of features of the 2nd and 3rd frame was obtained.

5)Relative scale between frames was obtained by taking the norm (mean distance ) between two clouds.

6)Now we update the rotation and translation wrt 1st frame.

7)Above steps are repeated for the whole dataset of images and we plot the translation(trajectory)wrt to 1st frame.


#### NOTE-Relative translation(translation wrt 1st frame) was calculated as we require global pose / ground truth for absolute translation.
