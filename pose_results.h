// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Header file for processing pose estimation results
 * @file pose_results.h
 */
#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include "openvino/openvino.hpp"

/**
 * @class PoseResult
 * @brief Class for processing pose estimation results
 * 
 * RTMPose model uses SimCC coordinate classification method.
 * Unlike traditional heatmap approaches, SimCC classifies x and y coordinates 
 * of each keypoint independently.
 * The output of the RTMPose model consists of two tensors:
 * 1. simcc_x: [batch_size, num_keypoints, width*simcc_split_ratio]
 * 2. simcc_y: [batch_size, num_keypoints, height*simcc_split_ratio]
 * 
 * Coordinate calculation method:
 * 1. Find the index with the largest value for each keypoint in simcc_x
 * 2. Find the index with the largest value for each keypoint in simcc_y
 * 3. Calculate the actual coordinates by dividing the found indices by simcc_split_ratio
 * 
 * simcc_split_ratio is typically set to 2.0 to increase the resolution of the model output.
 */
class PoseResult {
public:
    // Define pose keypoint structure
    struct Keypoint {
        float x, y;
        float confidence;
        int64_t label_id;
    };

private:
    const ov::Tensor _outputTensorX;  // simcc_x output tensor
    const ov::Tensor _outputTensorY;  // simcc_y output tensor
    const std::vector<std::string> _classNames;
    const std::vector<std::string> _imageNames;
    const float _confidenceThreshold;
    const float _simccSplitRatio;
    const float _inputWidth;
    const float _inputHeight;
    
    // Store processed keypoint results
    std::vector<Keypoint> _filtered_keypoints;

    // Function to extract keypoints
    std::vector<Keypoint> extractKeypoints() {
        const auto x_shape = _outputTensorX.get_shape();
        const auto y_shape = _outputTensorY.get_shape();
        const float* x_data = _outputTensorX.data<const float>();
        const float* y_data = _outputTensorY.data<const float>();
        
        std::vector<Keypoint> keypoints;
        
        // Output tensor shape: [batch_size, num_keypoints, extend_width/height]
        size_t num_keypoints = x_shape[1];
        size_t extend_width = x_shape[2];
        size_t extend_height = y_shape[2];
        
        // Process only the first batch of data
        for (size_t i = 0; i < num_keypoints; ++i) {
            Keypoint kp;
            
            // Find maximum value in x coordinate feature vector
            const float* x_features = x_data + i * extend_width;
            auto x_max_iter = std::max_element(x_features, x_features + extend_width);
            int x_max_pos = std::distance(x_features, x_max_iter);
            float x_score = *x_max_iter;
            
            // Find maximum value in y coordinate feature vector
            const float* y_features = y_data + i * extend_height;
            auto y_max_iter = std::max_element(y_features, y_features + extend_height);
            int y_max_pos = std::distance(y_features, y_max_iter);
            float y_score = *y_max_iter;
            
            // Calculate final coordinates (adjusted by split ratio)
            // In SimCC, the output feature map is typically larger than the input image size
            // For a 256x256 image, the output tensor might be 512x512
            // In this case, simcc_split_ratio = 2.0
            kp.x = x_max_pos / _simccSplitRatio;
            kp.y = y_max_pos / _simccSplitRatio;
            
            // Use the maximum of x and y scores as the confidence score
            kp.confidence = std::max(x_score, y_score);
            kp.label_id = i;  // Set label ID directly
            
            // Add to keypoints list regardless of confidence for debugging
            keypoints.push_back(kp);
        }
        
        // Filter keypoints by confidence and position
        std::vector<Keypoint> filtered;
        for (const auto& kp : keypoints) {
            bool is_valid = true;
            
            // 1. Check confidence score
            if (kp.confidence < _confidenceThreshold) {
                is_valid = false;
            }
            
            // 2. Check if coordinate is on image boundary
            float margin = 5.0f;  // Boundary margin
            
            if (kp.x < margin || kp.x > (_inputWidth - margin) || 
                kp.y < margin || kp.y > (_inputHeight - margin)) {
                is_valid = false;
            }
            
            // 3. Check for NaN
            if (std::isnan(kp.confidence)) {
                is_valid = false;
            }
            
            if (is_valid) {
                filtered.push_back(kp);
            }
        }
        
        return filtered;
    }

public:
    explicit PoseResult(const ov::Tensor& output_tensor_x,
                       const ov::Tensor& output_tensor_y,
                       const std::vector<std::string>& image_names = {},
                       float confidence_threshold = 0.05,
                       const std::vector<std::string>& class_names = {},
                       float simcc_split_ratio = 2.0f,
                       float input_width = 256.0f,
                       float input_height = 256.0f)
        : _outputTensorX(output_tensor_x),
          _outputTensorY(output_tensor_y),
          _classNames(class_names),
          _imageNames(image_names),
          _confidenceThreshold(confidence_threshold),
          _simccSplitRatio(simcc_split_ratio),
          _inputWidth(input_width),
          _inputHeight(input_height) {
              process();
          }

    /**
     * @brief Process results (extract keypoints)
     */
    void process() {
        _filtered_keypoints = extractKeypoints();
    }
    
    /**
     * @brief Get filtered keypoint results
     * @return Vector of filtered keypoints
     */
    const std::vector<Keypoint>& getFilteredKeypoints() const {
        return _filtered_keypoints;
    }
    
    /**
     * @brief Get class names
     * @return Vector of class names
     */
    const std::vector<std::string>& getClassNames() const {
        return _classNames;
    }
}; 