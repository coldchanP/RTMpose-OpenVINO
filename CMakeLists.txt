cmake_minimum_required(VERSION 3.16)

project(RTMPose_OpenVINO 
    VERSION 1.0.0
    DESCRIPTION "Real-time multi-person pose estimation using RTMPose and OpenVINO"
    LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    main.cpp
    rtmpose_openvino.cpp
    rtmdet_openvino.cpp
    rtmpose_tracker_openvino.cpp
    openvino_model_base.cpp
)

# Header files
set(HEADERS
    rtmpose_openvino.h
    rtmdet_openvino.h
    rtmpose_tracker_openvino.h
    rtmpose_utils.h
    openvino_model_base.h
    pose_results.h
    detection_results.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    openvino::runtime
)

# Compiler-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O3)
endif()

# Copy model files to build directory
file(GLOB MODEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/*")
if(MODEL_FILES)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/model 
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Copy video files to build directory
file(GLOB VIDEO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vid/*")
if(VIDEO_FILES)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/vid 
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Installation
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(DIRECTORY model/
    DESTINATION share/${PROJECT_NAME}/model
    FILES_MATCHING PATTERN "*.xml" PATTERN "*.bin"
)

# Print information
message(STATUS "RTMPose OpenVINO Configuration:")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}") 