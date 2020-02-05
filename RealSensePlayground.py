import numpy as np
import cv2
import pyrealsense2 as rs

# pipe = rs.pipeline()
# profile = pipe.start()
# try:
#   for i in range(0, 100):
#     frames = pipe.wait_for_frames()
#     for f in frames:
#       print(f.profile)
# finally:
#     pipe.stop()

isStreamDepthImg = True
isStreamColorImg = True
isStreamInfraredImg = True

cfg = rs.config()
if isStreamDepthImg:
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if isStreamColorImg:
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
if isStreamInfraredImg:
    cfg.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
#cfg.enable_stream(rs.stream.any) 

# TODO: figure out how to get the vision output -
#  cfg.enable_stream(rs.stream, 640, 480, rs.format.rgb8, 30)
# TypeError: enable_stream(): incompatible function arguments. The following argument types are supported:
#     1. (self: pyrealsense2.pyrealsense2.config, stream_type: pyrealsense2.pyrealsense2.stream, stream_index: int, width: int, height: int, format: pyrealsense2.pyrealsense2.format=format.any, framerate: int=0) -> None
#     2. (self: pyrealsense2.pyrealsense2.config, stream_type: pyrealsense2.pyrealsense2.stream, stream_index: int=-1) -> None
#     3. (self: pyrealsense2.pyrealsense2.config, stream_type: pyrealsense2.pyrealsense2.stream, format: pyrealsense2.pyrealsense2.format, framerate: int=0) -> None
#     4. (self: pyrealsense2.pyrealsense2.config, stream_type: pyrealsense2.pyrealsense2.stream, width: int, height: int, format: pyrealsense2.pyrealsense2.format=format.any, framerate: int=0) -> None
#     5. (self: pyrealsense2.pyrealsense2.config, stream_type: pyrealsense2.pyrealsense2.stream, stream_index: int, format: pyrealsense2.pyrealsense2.format, framerate: int=0) -> None

# Context and check the number of devices
rsContext = rs.context()
deviceList = rsContext.query_devices()
if deviceList.size() == 0:
    raise Exception('No device detected. Is it plugged in?')

# for device in deviceList:
device = deviceList.front()
device.get_info(rs.camera_info.name)

pipe = rs.pipeline()
pipelineProfile = pipe.start(cfg)

try:
    #for i in range(0, 300):
    while True:

        # Plot the depth frame
        frames = pipe.wait_for_frames()
        # frames.size()
        # frame = frames.as_frame()

        if isStreamColorImg:
            frame = frames.get_color_frame()
            depth_data = frame.get_data()
            np_image = np.asanyarray(depth_data)
            np.shape(np_image)
            np_imageFloat = np_image.astype(float)
            np_imageFloat = np_imageFloat / np.max(np_imageFloat)
            cv2.namedWindow("Color Image", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Color Image", np_imageFloat)
            cv2.waitKey(30)

        if isStreamInfraredImg:
            frame = frames.get_infrared_frame()
            depth_data = frame.get_data()
            np_image = np.asanyarray(depth_data)
            cv2.equalizeHist(np_image, np_image)
            np_image = cv2.applyColorMap(np_image, cv2.COLORMAP_JET)
            cv2.imshow("Infrared Image", np_image)
            cv2.namedWindow("Infrared Image", cv2.WINDOW_AUTOSIZE)
            cv2.waitKey(30)

        if isStreamDepthImg:
            frame = frames.get_depth_frame()
            depth_data = frame.get_data()
            np_image = np.asanyarray(depth_data)
            if False:
                np.max(np_image)
                hist, bin_edges =  np.histogram(np_image, bins=150)
                import matplotlib.pyplot as plt
                _ = plt.hist(hist, bins=bin_edges)
                plt.show()

            norm_image = cv2.normalize(np_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.equalizeHist(norm_image, norm_image)
            norm_image = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
            cv2.imshow("Depth Image (disparity)", norm_image)
            cv2.waitKey(30)
    # print(f.profile)
finally:
    pipe.stop()

if False:
    frames = pipe.wait_for_frames()
    # frames.size()
    #frame = frames.as_frame()

    frame = frames.get_depth_frame()


    depth_data = frame.get_data()
    np_image = np.asanyarray(depth_data)
    np.shape(np_image)
    np_imageFloat = np_image.astype(float)
    np_imageFloat = np_imageFloat / np.max(np_imageFloat)


    # Convert an image from numpy to opencv
    # new_image = np.ndarray((3, num_rows, num_cols), dtype=int)

    #Displayed the image
    cv2.imshow("WindowNameHere", np_imageFloat)
    cv2.waitKey(30)


    depth = frames.get_depth_frame()
    depth_data = depth.as_frame().get_data()




# try:
#   for i in range(0, 100):
#     frames = pipe.wait_for_frames()
#     for f in frames:
#       print(f.profile)
# finally:
#     pipe.stop()



rs.device_list


sensorsList = rsContext.query_all_sensors






