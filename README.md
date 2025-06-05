# RTMPose OpenVINO

A real-time pose estimation application using RTMPose models with OpenVINO optimization for efficient inference on CPU/GPU.

## Features

- **Multi-mode pose estimation**: Support for body, face, and hand pose detection
- **Real-time performance**: Optimized with OpenVINO for fast inference
- **Cross-platform**: Compatible with Windows, Linux, and macOS
- **Flexible input**: Support for webcam, video files, and image sequences
- **Tracking capability**: Built-in pose tracking for smooth temporal consistency

## Requirements

### System Requirements
- OpenVINO Toolkit 2023.0 or later
- OpenCV 4.5.0 or later
- Visual Studio 2019 or later

### Hardware Requirements
- CPU: Intel Core i5 or equivalent (recommended)
- RAM: 8GB minimum, 16GB recommended
- GPU: Intel integrated graphics or discrete GPU (optional, for GPU acceleration)

## Setup Instructions

### 1. Prepare Third-party Libraries

Create a `thirdparty` folder in the project root and place the following libraries:

```
thirdparty/
├── openvino/           # OpenVINO runtime libraries
└── opencv/             # OpenCV libraries
```

**OpenVINO Setup:**
- Download and install OpenVINO Toolkit from [Intel's official website](https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html)
- Copy the runtime libraries to `thirdparty/openvino/`

**OpenCV Setup:**
- Download pre-built OpenCV binaries or build from source
- Copy the libraries to `thirdparty/opencv/`

### 2. Prepare Model Files

Create a `model` folder in the project root and place the required model files:

```
model/
├── face_det.xml        # Face detection model (XML format)
├── face_det.bin        # Face detection model (binary weights)
├── face_pose.xml       # Face pose estimation model (XML format)
├── face_pose.bin       # Face pose estimation model (binary weights)
├── body_det.xml        # Body detection model (XML format)
├── body_det.bin        # Body detection model (binary weights)
├── body_pose.xml       # Body pose estimation model (XML format)
├── body_pose.bin       # Body pose estimation model (binary weights)
├── hand_det.xml        # Hand detection model (XML format)
├── hand_det.bin        # Hand detection model (binary weights)
├── hand_pose.xml       # Hand pose estimation model (XML format)
└── hand_pose.bin       # Hand pose estimation model (binary weights)
```

**Model Requirements:**
- **Face models**: For facial landmark detection (68 or 106 keypoints)
- **Body models**: For human body pose estimation (17 or 133 keypoints)
- **Hand models**: For hand keypoint detection (21 keypoints)

**Note**: You need to convert RTMPose models to OpenVINO IR format (.xml/.bin) using the OpenVINO Model Optimizer or export them directly from the original framework.

### 3. Build the Project

#### Using Visual Studio

1. Open `RTMpose_openvino.sln` in Visual Studio
2. Set the configuration to Release
3. Build the solution (Ctrl+Shift+B)

## Usage

### Basic Usage

```bash
# Run with default settings (face mode, webcam input)
./RTMpose_openvino

# The application will automatically:
# 1. Load the specified models
# 2. Initialize the camera
# 3. Start real-time pose estimation
```

### Configuration

Modify the following parameters in `main.cpp`:

```cpp
// Model paths
std::string rtm_detnano_xml_path = "./model/face_det.xml";
std::string rtm_detnano_bin_path = "./model/face_det.bin";
std::string rtm_pose_xml_path = "./model/face_pose.xml";
std::string rtm_pose_bin_path = "./model/face_pose.bin";

// Mode selection: 0: BODY, 1: FACE, 2: HAND
int mode = 1;

// Pose confidence threshold
float pose_thres = 0.2;

// Device selection: "CPU", "GPU", "AUTO"
std::string device = "CPU";
```

### Key Controls

- Press `ESC` or `q` to quit the application
- Real-time FPS information is displayed in the console

## Project Structure

```
RTMpose_openvino/
├── main.cpp                           # Main application entry point
├── rtmpose_tracker_openvino.cpp       # Main pose tracking implementation
├── rtmpose_tracker_openvino.h         # Pose tracker header
├── rtmpose_openvino.cpp               # RTMPose model wrapper
├── rtmpose_openvino.h                 # RTMPose header
├── rtmdet_openvino.cpp                # Detection model implementation
├── rtmdet_openvino.h                  # Detection header
├── openvino_model_base.cpp            # Base OpenVINO model class
├── openvino_model_base.h              # Base model header
├── rtmpose_utils.h                    # Utility functions
├── pose_results.h                     # Pose result structures
├── detection_results.h                # Detection result structures
├── RTMpose_openvino.sln               # Visual Studio solution file
├── RTMpose_openvino.vcxproj           # Visual Studio project file
├── model/                             # Model files directory
├── thirdparty/                        # Third-party libraries
└── README.md                          # This file
```

## Performance Optimization

- **CPU Optimization**: Use Intel MKL-DNN for better CPU performance
- **GPU Acceleration**: Enable OpenVINO GPU plugin for Intel Graphics
- **Memory Management**: Pre-allocated buffers to reduce allocation overhead
- **Threading**: Optimized OpenCV threading for better performance

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure model files are in the correct format and paths
2. **Camera Access Error**: Check camera permissions and availability
3. **Performance Issues**: Try reducing input resolution or using CPU-only mode
4. **Build Errors**: Verify OpenVINO and OpenCV installations

### Debug Mode

To enable verbose logging:

1. In Visual Studio, set the configuration to Debug
2. Build and run the solution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Author**: SC Park
**Email**: tjdcks7570@kw.ac.kr

## Acknowledgments

- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) - Original RTMPose implementation
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel's inference optimization toolkit
- [OpenCV](https://opencv.org/) - Computer vision library


