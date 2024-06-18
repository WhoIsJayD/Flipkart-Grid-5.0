import numpy as np
import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 0, 320, 240, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 0, 640,480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    # Wait for a few frames to allow auto-exposure, etc., to settle
    for _ in range(30):
        pipeline.wait_for_frames()

    # Get the first available frame
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Get the intrinsics of the depth and color cameras
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    # Get the extrinsics between the depth and color cameras
    extrinsics_dc = depth_frame.profile.get_extrinsics_to(color_frame.profile)

    # Print the intrinsic parameters
    print("Depth Intrinsics:")
    print(f"Width: {depth_intrinsics.width}, Height: {depth_intrinsics.height}")
    print(f"FX: {depth_intrinsics.fx}, FY: {depth_intrinsics.fy}")
    print(f"CX: {depth_intrinsics.ppx}, CY: {depth_intrinsics.ppy}")
    print("Color Intrinsics:")
    print(f"Width: {color_intrinsics.width}, Height: {color_intrinsics.height}")
    print(f"FX: {color_intrinsics.fx}, FY: {color_intrinsics.fy}")
    print(f"CX: {color_intrinsics.ppx}, CY: {color_intrinsics.ppy}")

    # Print the extrinsic parameters
    print("Extrinsics between Depth and Color cameras:")
    print("Rotation Matrix:")
    print(np.array(extrinsics_dc.rotation).reshape(3, 3))
    print("Translation Vector:")
    print(np.array(extrinsics_dc.translation))

finally:
    # Stop the pipeline and release resources
    pipeline.stop()
