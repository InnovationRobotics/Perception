import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt



def main():

    # extact features from cropped image of the object
    refImagePath = '/home/sload/Desktop/ShaharSarShalom/Perception/VehicleTopView/VehcleTopView.bmp'

    refImg = cv.imread(refImagePath)
    # cv2.imshow("", refImg)

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

    # Find homography  findHomography(srcPoints, dstPoints, method=None, ransacReprojThreshold=None, mask=None, maxIters=None, confidence=None)
    # TODO: use affine ransac instead of homography ransac implementation (because later on we estimate affine transformation only for the inliers)
    h, mask = cv.findHomography(srcPoints=pointsCur, dstPoints=pointsRef, method=cv.RANSAC, ransacReprojThreshold=int(3), confidence=0.9)
    # h, mask = cv.findHomography(srcPoints=pointsCur, dstPoints=pointsRef, method=cv.RANSAC, ransacReprojThreshold=int(3), mask=None, maxIters=None, confidence=0.9)

    # Estimate affine transform only from the inliers points that surpassed ransac

    # Use the inliers in order to


    # def findHomography(srcPoints, dstPoints, method=None, ransacReprojThreshold=None, mask=None, maxIters=None,
    #                    confidence=None):  # real signature unknown; restored from __doc__
    # cv.find

    cv2.xfeatures2d.SURF_create()


    cv2.xfeatures2d.



if __name__ == '__main__':
    sys.exit(main() or 0)