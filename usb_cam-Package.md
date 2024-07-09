# USB Cam ROS Package Documentation

### Pixel Formats

The supported pixel formats and associated utilities can be found in:
- `src/usb_cam/include/usb_cam/formats`
- Formats include: `av`, `m240`, `mjpeg`, `mono`, `rgb`, `utils`, `uyvy`, `yuyv`

## Package Structure

- **Package Manifest (`package.xml`)**: Contains metadata including the package name, version, dependencies, maintainers.

- **Build Configuration (`CMakeLists.txt`)**: Specifies how the package should be built, including executable targets, package dependencies, and installation rules.

- **Resource Files**: Includes calibration files and other necessary resources, `config` folder.

## Launching the Executable

When the `usb_cam` executable is launched, it initializes and runs several key components outlined in the package structure:

- **Main Node File (`usb_cam_node.cpp`)**: This C++ file contains the core functionality of the camera node. Upon execution, it initializes the ROS2 node, manages communication with the USB camera, captures images, and publishes image data. This file directly interacts with the camera hardware through the commands specified in the other source files.

- **Camera Driver File (`usb_cam.cpp`)**: Engaged by the main node file to handle lower-level interactions with the camera hardware, such as opening the camera device, configuring camera settings, capturing frames, and error management.

### Configuration and Calibration

- **Configuration Files**: Parameters like video device path, pixel format, and frame rate are loaded from YAML files specified in the launch scripts. 

- **Calibration Files**: Contains calibration data that corrects for distortions and aligns the camera output to real-world dimensions.

## Communication

As the node operates, it communicates via the following ROS topics:

- **`/image_raw`**: Publishes uncompressed image data, showing raw output from the camera.
- **`/camera_info`**: Provides metadata about the camera configuration, including calibration data.
- **`/compressed_image`**: Streams compressed image formats for bandwidth-efficient transmission.
