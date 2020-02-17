import numpy as np
import cv2
from pylab import *
import open3d as o3d
import pyrealsense2 as rs
import os

def CheckFrameValidity(frame):
    return frame.is_depth_frame() or frame.is_frame() or frame.is_frameset() or frame.is_motion_frame() or frame.is_points() or frame.is_pose_frame() or frame.is_video_frame()

def PlotDepthFrame(depthFrame):
    assert isinstance(depthFrame, rs.depth_frame) or isinstance(depthFrame, rs.frame)

    depth_data = depthFrame.get_data()
    np_image = np.asanyarray(depth_data)
    if False:
        np.max(np_image)
        hist, bin_edges = np.histogram(np_image, bins=150)
        import matplotlib.pyplot as plt
        _ = plt.hist(hist, bins=bin_edges)
        plt.show()

    norm_image = cv2.normalize(np_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.equalizeHist(norm_image, norm_image)
    norm_image = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
    cv2.imshow("Depth Image (disparity)", norm_image)
    cv2.waitKey(30)

def PlotVideoFrame(videoFrame):
    assert isinstance(videoFrame, rs.video_frame)

    videoFrameData = videoFrame.get_data()
    np_image = np.asanyarray(videoFrameData)
    np_imageRGB = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    cv2.namedWindow("Color Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Color Image", np_imageRGB)
    cv2.waitKey(30)

def PlotPointCloud(pc, depthFrame, videoFrame):
    # Map the point cloud to the given color frame
    pc.map_to(videoFrame)
    # Generate the pointcloud and texture mappings of depth map.
    points = pc.calculate(depthFrame)

    # This is the point cloud for each pixel in the depth image
    pointsVertices = points.get_vertices()
    pointNp = np.asanyarray(pointsVertices)

    # Retrieve the texture coordinates (uv map) for the point cloud
    tex_coords = points.get_texture_coordinates()
    # tex_coordsNp = np.asanyarray(tex_coords)

    pcList = []
    i = 0
    for itr in np.nditer(pointNp):
        x = itr.item()
        # Print only points with depth
        if x[2] != 0:
            pcList.append(np.array([x[0], x[1], x[2]], np.float))

    pcArr = np.stack(pcList, axis=0)
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcArr)
    o3d.visualization.draw_geometries([pcd])

def WriteVideoFrame(videoFrame, dstFolder, fileIndex):
    # write a video frame input to the file system
    assert isinstance(videoFrame, rs.video_frame)

    videoFrameData = videoFrame.get_data()
    np_image = np.asanyarray(videoFrameData)
    np_imageRGB = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(dstFolder, 'Frame_' + '{num:06}'.format(num=fileIndex) + '.bmp'), np_imageRGB)

def ApplyFiltersOnDepthFrame(depthFrame):
    # Apply filters.
    # The implemented flow of the filters pipeline is in the following order:
    # 1. apply decimation filter
    # 2. transform the scence into disparity domain
    # 3. apply spatial filter
    # 4. apply temporal filter
    # 5. revert the results back (if step Disparity filter was applied
    # to depth domain (each post processing block is optional and can be applied independantly).
    revert_disparity = True;

    # Decimation - reduces depth frame density
    decimateFilter = rs.decimation_filter()
    # Converts from depth representation to disparity representation and vice - versa in depth frames
    depth_to_disparity = rs.disparity_transform(True)
    # Spatial    - edge-preserving spatial smoothing
    spatial_filter = rs.spatial_filter()
    # Temporal   - reduces temporal noise
    temporalFilter = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)

    # Apply all filters
    filteredFrame = decimateFilter.process(depthFrame)
    filteredFrame = depth_to_disparity.process(filteredFrame)
    filteredFrame = spatial_filter.process(filteredFrame)
    filteredFrame = temporalFilter.process(filteredFrame)
    filteredFrame = disparity_to_depth.process(filteredFrame)

    # PlotDepthFrame(filteredFrame)
    # PlotDepthFrame(depthFrame)
    return filteredFrame

def FramesToVideo(srcFolder = '/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/20200209_163301/DebugPath/2020_02_17_08_35_12', fps = 10):
    # The function create a video file from all the frames in the source folder
    #srcFolder = '/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/20200209_163301/DebugPath/2020_02_17_08_35_12'
    # Frames per second
    fps = 10

    videoWriter = None

    ld = os.listdir(srcFolder)
    ld.sort()
    for filename in ld:
        if filename.endswith(".bmp") or filename.endswith(".jpg"):
            print(os.path.join(srcFolder, filename))
            img = cv2.imread(os.path.join(srcFolder, filename))
            # cv.imshow("", curImg)

            if videoWriter is None:
                height, width, layers = img.shape
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                videoWriter = cv2.VideoWriter(os.path.join(srcFolder, 'video.avi'), fourcc, fps, (width, height))

            videoWriter.write(img)

    if videoWriter is not None:
        videoWriter.release()

def main():
    rsContext = rs.context()

    folderPath = '/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/'
    #recordedFile = os.path.join(folderPath, '20200209_163025.bag')
    recordedFile = os.path.join(folderPath, '20200209_163301.bag')
    dstFolder = '/home/sload/Desktop/ShaharSarShalom/VideoStreamSamples/20200209_163301/VideoFrames'

    device = rsContext.load_device(recordedFile)

    device = device.as_playback()

    pipe = rs.pipeline(rsContext)
    pipelineProfile = pipe.start()

    isStreamColorImg = True
    isStreamInfraredImg = True
    isStreamDepthImg = True
    isStreamPointCloud = True

    # Declare a point cloud object
    pc = rs.pointcloud()

    #
    # def map_to(self, mapped):  # real signature unknown; restored from __doc__
    #     """
    #     map_to(self: pyrealsense2.pyrealsense2.pointcloud, mapped: pyrealsense2.pyrealsense2.frame) -> None
    #
    #     Map the point cloud to the given color frame.
    #     """
    #     pass


    try:
        #for i in range(0, 300):
        videoFrameCounter = 0
        while True:

            # Wait for all configured streams to produce a frame
            frames = pipe.wait_for_frames()
            # frames.size()
            # frame = frames.as_frame()

            if isStreamColorImg:
                videoFrameCounter += 1
                videoFrame = frames.get_color_frame()

                if videoFrame.is_frame() and CheckFrameValidity(videoFrame):
                    if False:
                        PlotVideoFrame(videoFrame)
                    # WriteVideoFrame(videoFrame = videoFrame, dstFolder = dstFolder, fileIndex= videoFrameCounter)

            if isStreamInfraredImg:
                frame = frames.get_infrared_frame()
                if CheckFrameValidity(frame):
                    depth_data = frame.get_data()
                    np_image = np.asanyarray(depth_data)
                    cv2.equalizeHist(np_image, np_image)
                    np_image = cv2.applyColorMap(np_image, cv2.COLORMAP_JET)
                    cv2.imshow("Infrared Image", np_image)
                    cv2.namedWindow("Infrared Image", cv2.WINDOW_AUTOSIZE)
                    cv2.waitKey(30)

            if isStreamDepthImg:
                depthFrame = frames.get_depth_frame()
                if CheckFrameValidity(depthFrame):
                    PlotDepthFrame(depthFrame)
                    # Plot the depth map as a point cloud
                    # PlotPointCloud(pc, depthFrame, videoFrame)

                    depthFrameFiltered = ApplyFiltersOnDepthFrame(depthFrame)
                    PlotDepthFrame(depthFrame)

        # print(f.profile)
        device.resume()
    finally:
        pipe.stop()
    # pyrealsense2.pyrealsense2.playback
    # //Create a context
    # rs2::context ctx;
    # //Load the recorded file to the context
    # rs2::playback device = ctx.load_device("my_file_name.bag");
    # //playback "is a" device, so just use it like any other device now

if __name__ == '__main__':
    sys.exit(main() or 0)