import sys
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


class Local:
    def __init__(self):
        self.

    def LoadRef(self, ):
        # Load the feature to locate every frame




def main():

    # extact features from cropped image of the object
    refImagePath = '/home/sload/Desktop/ShaharSarShalom/Perception/VehicleTopView/VehcleTopView.bmp'

    refImg = cv.imread(refImagePath)
    # cv.imshow("", refImg)

    # directory = '/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/20200209_163301/VideoFrames/'
    # for filename in os.listdir(directory):
    #     if filename.endswith(".bmp") or filename.endswith(".py"):
    #         # print(os.path.join(directory, filename))
    #         continue
    #     else:
    #         continue

    curImagePath = '/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/20200209_163301/VideoFrames/Frame_000100.bmp'
    curImg = cv.imread(curImagePath)
    # cv2.imshow("", curImg)

    # Initiate detector
    # Original code comes from here: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    #detectorObj = cv2.xfeatures2d.SURF_create()
    detectorObj = cv.ORB_create()

    keypointsRef, descriptorsRef = detectorObj.detectAndCompute(refImg, None)
    keypointsCur, descriptorsCur = detectorObj.detectAndCompute(curImg, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(descriptorsRef, descriptorsCur)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    matchedKeyPointsImg = cv.drawMatches(refImg, keypointsRef, curImg, keypointsCur, matches[:15], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matchedKeyPointsImg), plt.show()

    # Extract location of good matches
    pointsRef = np.zeros((len(matches), 2), dtype=np.float32)
    pointsCur = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        pointsRef[i, :] = keypointsRef[match.queryIdx].pt
        pointsCur[i, :] = keypointsCur[match.trainIdx].pt

    # Note: This function calculate only 4 degrees of freedom !!! scaling, rotation and translation
    h, mask = cv.estimateAffinePartial2D(pointsRef, pointsCur, method=cv.RANSAC, ransacReprojThreshold=3, confidence=0.9)

    # Estimate the rotation angle from the matrix [s]
    # Extract traslation
    dx = h[0, 2]
    dy = h[1, 2]

    # Extract rotation angle
    da = np.arctan2(h[1, 0], h[0, 0])
    # print(np.rad2deg(da))

    # Store transformation
    transforms = [dx, dy, da]

    # Plot a bounding box that represent the object in the current image
    refBoundingBox = np.array([ [0, 0], [0,refImg.shape[0]-1], [refImg.shape[1]-1,refImg.shape[0]-1], [refImg.shape[1]-1,0]])

    # reshape pointsIn from numLandmarks x 2 to numLandmarks x 1 x 2
    tt = np.reshape(refBoundingBox, (refBoundingBox.shape[0], 1, refBoundingBox.shape[1]))
    refBoundingBoxTransformedToCurImg = cv.transform(tt, h)

    curImgWithRoi = cv.polylines(img=curImg.copy(), pts=[refBoundingBoxTransformedToCurImg], isClosed=True, color=(0,0,255),thickness=3)
    cv.imshow("", curImgWithRoi)

    img = cv.polylines(img=curImg, pts=np.array([[1,1], [200,200]]), isClosed=False, color=(0, 0, 255),thickness=3)

if __name__ == '__main__':
    sys.exit(main() or 0)