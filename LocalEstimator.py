import sys
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from datetime import datetime
from argparse import ArgumentParser


class LocalEstimator:
    def __init__(self, debugFlag = False, debugPath = ''):

        # Original code comes from here: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        # detectorObj = cv2.xfeatures2d.SURF_create()
        self.detectorObj = cv.ORB_create()
        self.debugFlag = debugFlag
        if self.debugFlag:
            # Create a new folder for debug purposes
            now = datetime.now()

            dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
            self.debugPath = os.path.join(debugPath, dt_string)
            try:
                os.mkdir(self.debugPath)
            except OSError as error:
                print(error)

                # Debug frame counter
            self.debugFrameCounter = 0

    def LoadRef(self, refImg):
        # Load the feature to locate every frame
        self.keypointsRef, self.descriptorsRef = self.detectorObj.detectAndCompute(refImg, None)
        if self.debugFlag:
            self.refImg = refImg

    def CalculateLocal(self, curImg):
        keypointsCur, descriptorsCur = self.detectorObj.detectAndCompute(curImg, None)

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(self.descriptorsRef, descriptorsCur)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        if False and self.debugFlag:
            # Draw first 10 matches.
            matchedKeyPointsImg = cv.drawMatches(self.refImg, self.keypointsRef, curImg, keypointsCur, matches[:15], None,
                                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(matchedKeyPointsImg), plt.show()

        # Extract location of good matches
        pointsRef = np.zeros((len(matches), 2), dtype=np.float32)
        pointsCur = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            pointsRef[i, :] = self.keypointsRef[match.queryIdx].pt
            pointsCur[i, :] = keypointsCur[match.trainIdx].pt

        # Note: This function calculate only 4 degrees of freedom !!! scaling, rotation and translation
        h, mask = cv.estimateAffinePartial2D(pointsRef, pointsCur, method=cv.RANSAC, ransacReprojThreshold=3,confidence=0.9)

        # Estimate the rotation angle from the matrix [s]
        # Extract traslation
        dx = h[0, 2]
        dy = h[1, 2]

        # Extract rotation angle
        da = np.rad2deg(np.arctan2(h[1, 0], h[0, 0]))
        # print(da)

        # Store transformation
        trans = [dx, dy, da]

        # Plot a bounding box that represent the object in the current image
        if self.debugFlag:
            refBoundingBox = np.array(
                [[0, 0], [0, self.refImg.shape[0] - 1], [self.refImg.shape[1] - 1, self.refImg.shape[0] - 1], [self.refImg.shape[1] - 1, 0]])

            # reshape pointsIn from numLandmarks x 2 to numLandmarks x 1 x 2
            refBoundingBoxShaped = np.reshape(refBoundingBox, (refBoundingBox.shape[0], 1, refBoundingBox.shape[1]))
            refBoundingBoxTransformedToCurImg = cv.transform(refBoundingBoxShaped, h)

            curImgWithRoi = cv.polylines(img=curImg.copy(), pts=[refBoundingBoxTransformedToCurImg], isClosed=True,
                                         color=(0, 0, 255), thickness=3)

            # Plot a cross at the center of the blob
            offset = 10
            refBoundingBox = np.array(
                [[self.refImg.shape[1]/2 - offset, self.refImg.shape[0]/2], [self.refImg.shape[1]/2 + offset, self.refImg.shape[0]/2]])
            refBoundingBoxShaped = np.reshape(refBoundingBox, (refBoundingBox.shape[0], 1, refBoundingBox.shape[1]))
            refBoundingBoxTransformedToCurImg = cv.transform(refBoundingBoxShaped, h).astype('int')
            curImgWithRoi = cv.polylines(img=curImgWithRoi, pts=[refBoundingBoxTransformedToCurImg], isClosed=False,
                                         color=(0, 0, 255), thickness=3)
            refBoundingBox = np.array(
                [[self.refImg.shape[1]/2, self.refImg.shape[0]/2 - offset], [self.refImg.shape[1]/2, self.refImg.shape[0]/2 + offset]] )
            refBoundingBoxShaped = np.reshape(refBoundingBox, (refBoundingBox.shape[0], 1, refBoundingBox.shape[1]))
            refBoundingBoxTransformedToCurImg = cv.transform(refBoundingBoxShaped, h).astype('int')
            curImgWithRoi = cv.polylines(img=curImgWithRoi, pts=[refBoundingBoxTransformedToCurImg], isClosed=False,
                                         color=(0, 0, 255), thickness=3)

            # cv.imshow("", curImgWithRoi)
            st = 'dx {num:.2f} '.format(num=trans[0]) + 'dy {num:.2f} '.format(num=trans[1]) + 'theta {num:.2f}'.format(num=trans[2])
            # '{num:.2f}'.format(num=trans[0])
            cv.putText(curImgWithRoi, st, (32, 32), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv.imshow("", curImgWithRoi)

            cv.imwrite(os.path.join(self.debugPath, '{num:06}'.format(num=self.debugFrameCounter)+'.jpg'), curImgWithRoi)
            self.debugFrameCounter += 1

        return trans


class LocalUnitTest:

    def Test1(self, config):
        # Test the ref to current matching frame to frame

        # Extact features from cropped image of the object
        refImagePath = '/home/sload/Desktop/ShaharSarShalom/Perception/VehicleTopView/VehcleTopView.bmp'
        refImg = cv.imread(refImagePath)
        # cv.imshow("", refImg)

        curImagePath = '/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/20200209_163301/VideoFrames/Frame_000100.bmp'
        curImg = cv.imread(curImagePath)
        # cv2.imshow("", curImg)

        localEstimator = LocalEstimator(config.isDebug)
        localEstimator.LoadRef(refImg)
        trans = localEstimator.CalculateLocal(curImg=curImg)

    def Test2(self, config):
        # Test the local matching module along time
        localEstimator = LocalEstimator(config.isDebug, config.debugPath)

        # Extact features from cropped image of the object
        refImagePath = '/home/sload/Desktop/ShaharSarShalom/Perception/VehicleTopView/VehcleTopView.bmp'
        refImg = cv.imread(refImagePath)
        # cv.imshow("", refImg)
        localEstimator.LoadRef(refImg=refImg)

        srcDir = '/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/20200209_163301/VideoFrames/'
        ld = os.listdir(srcDir)
        ld.sort()
        for filename in ld:
            if filename.endswith(".bmp"):
                print(os.path.join(srcDir, filename))
                curImg = cv.imread(os.path.join(srcDir, filename))
                # cv.imshow("", curImg)
                trans = localEstimator.CalculateLocal(curImg=curImg)


def main():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-isDebug', '--isDebug', default=True, help='')
    args.add_argument('-debugPath', '--debugPath', default='/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/20200209_163301/DebugPath', help='')
    args = parser.parse_args()

    localUnitTest = LocalUnitTest()
    # localUnitTest.Test1(config=args)
    localUnitTest.Test2(config=args)

    print('Finished unit test')


if __name__ == '__main__':
    sys.exit(main() or 0)