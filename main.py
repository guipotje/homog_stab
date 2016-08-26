#	Simple homography-based video stabilization
#	Author: Guilherme Potje



import cv2
import numpy as np

import Tkinter as tk
from Tkinter import *
from tkFileDialog import askopenfilename


#Parameters
WINDOW_SIZE = 15 
skip = 1 # speedup -- set 1 for original speed
resize = 0.5 #scale video resolution


def find_homography(kp1, des1, kp2, des2):

	bf = cv2.BFMatcher(cv2.NORM_L2)

	# Match descriptors.
	matches = bf.knnMatch(des1,des2,k=2)

	# Apply ratio test
	good = []
	for m,n in matches:
	    if m.distance < 0.9*n.distance:
	       good.append(m)


	pts1 = []
	pts2 = []

	for elem in good:
		pts1.append(kp1[elem.queryIdx].pt)
		pts2.append(kp2[elem.trainIdx].pt)

	pts1 = np.array(pts1)
	pts2 = np.array(pts2)

	M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)

	count_inliers = np.count_nonzero(mask)

	print 'Number of inliers: ', np.count_nonzero(mask)

	return count_inliers, M



root = tk.Tk()
root.withdraw()
file = askopenfilename()
root.destroy()

if len(file) == 0:
	print 'please select a video.'
	quit()

cap = cv2.VideoCapture(file)

frames = []
mean_homographies = []
median_homographies = []
corrected_frames = []
i = 0

while True: #and i<400:
	if cap.grab():
		flag, frame = cap.retrieve()
		if not flag:
			continue
		elif i%skip == 0:
			frame = cv2.resize(frame, (0,0), fx=resize, fy=resize)
            #cv2.imshow('video', frame)
			frames.append(frame)
		i+=1
	else:
		break
    #cv2.waitKey(1)
  
fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
size_orig = (frames[0].shape[1], frames[0].shape[0])
out_orig =  cv2.VideoWriter(file+'__original.avi',fourcc,30.0,size_orig)#cv2.VideoWriter('stab.mp4',-1, 30.0, (frames[0].shape[0], frames[0].shape[1]))

#write original video with skip x speedup
for frame in frames:
	out_orig.write(frame)

out_orig.release()

orb = cv2.SIFT(nfeatures=1000)

vec_kps = []
vec_descs = []

print 'extracting keypoints...'

for i in range(len(frames)):
	# find the keypoints and descriptors 
	kp1, des1 = orb.detectAndCompute(frames[i],None)

	vec_kps.append(kp1)
	vec_descs.append(des1)

	print 'Frame %d/%d: found %d keypoints'% (i,len(frames),len(kp1))



for i in range(len(frames)):
	mean_H = np.zeros((3,3), dtype='float64')
	median_H = []
	mean_C = 0
	median_vals = []
	k =  int(WINDOW_SIZE/2.0)+1
	for j in range(1,k,1): #for each couple neighbor frames iterated by distance
		if i-j >= 0 and i+j < len(frames):
			inliers_c, H = find_homography(vec_kps[i],vec_descs[i], vec_kps[i-j], vec_descs[i-j])
			inliers_c2, H2 = find_homography(vec_kps[i],vec_descs[i], vec_kps[i+j], vec_descs[i+j])
			print 'pair (%d,%d) has %d inliers'% (i,i-j,inliers_c)
			print 'pair (%d,%d) has %d inliers'% (i,i+j,inliers_c2)
			if inliers_c > 80 and inliers_c2 > 80: #ensures that neighbors are equally selected by distance to correctly balance the homography
				mean_H = mean_H + H
				mean_H = mean_H + H2
				mean_C+=2

	if mean_C > 0:
		mean_homographies.append(mean_H/mean_C) # Mean homography
	else:
		mean_homographies.append(np.eye(3, dtype='float64'))
	
	#print mean_H/mean_C
	#print median_vals
	#raw_input()

    #fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
    #fourcc = cv2.cv.CV_FOURCC('R','G','B',' ')
    #fourcc = cv2.cv.CV_FOURCC('Y','U','Y','2')
    #fourcc = cv2.cv.CV_FOURCC('Y','U','Y','U')
    #fourcc = cv2.cv.CV_FOURCC('U','Y','V','Y')
    #fourcc = cv2.cv.CV_FOURCC('I','4','2','0')
    #fourcc = cv2.cv.CV_FOURCC('I','Y','U','V')
    #fourcc = cv2.cv.CV_FOURCC('Y','U','1','2')
    #fourcc = cv2.cv.CV_FOURCC('Y','8','0','0')
    #fourcc = cv2.cv.CV_FOURCC('G','R','E','Y')
    #fourcc = cv2.cv.CV_FOURCC('B','Y','8',' ')
    #fourcc = cv2.cv.CV_FOURCC('Y','1','6',' ')
fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
    #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
#fourcc = cv2.cv.CV_FOURCC('M','P','E','G')

crop_x = 80
crop_y = 60

size = (frames[0].shape[1]-crop_x*2, frames[0].shape[0]-crop_y*2)
out =  cv2.VideoWriter(file+'__estabilizado.avi',fourcc,30.0,size)#cv2.VideoWriter('stab.mp4',-1, 30.0, (frames[0].shape[0], frames[0].shape[1]))

for i in range(len(frames)):
	corrected = cv2.warpPerspective(frames[i],mean_homographies[i],(0,0))
	#cv2.imshow('video corrected', corrected)
	#cv2.waitKey(10)
	out.write(corrected[crop_y:frames[0].shape[0]-crop_y, crop_x:frames[0].shape[1]-crop_x])

#print corrected.shape

cap.release()
out.release()

