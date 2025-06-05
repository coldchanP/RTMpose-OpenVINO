// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with output detection results
 * @file detection_results.h
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
 * @class DetectionResult
 * @brief A DetectionResult creates an output table with detection results
 */
class DetectionResult {
public:
    // Bounding box structure definition - moved to public for external access
    struct BoundingBox {
        float x_min, y_min, x_max, y_max;
        float confidence;
        int64_t label_id;
        size_t index;

        bool operator<(const BoundingBox& other) const {
            return confidence > other.confidence; // Sort in descending order by confidence
        }
    };

private:
    const std::string _idStr = "id";
    const std::string _labelStr = "label";
    const std::string _confidenceStr = "confidence";
    const std::string _xminStr = "xmin";
    const std::string _yminStr = "ymin";
    const std::string _xmaxStr = "xmax";
    const std::string _ymaxStr = "ymax";
    
    const ov::Tensor _detsTensor;
    const ov::Tensor _labelsTensor;
    const std::vector<std::string> _classNames;
    const std::vector<std::string> _imageNames;
    const float _confidenceThreshold;
    const float _iouThreshold;
    const size_t _topK;  // Show only top K results
    
    // Variable to store results after NMS
    std::vector<BoundingBox> _filtered_boxes;

    // Function to calculate IoU (Intersection over Union)
    float calculateIoU(const BoundingBox& box1, const BoundingBox& box2) const {
        // Calculate intersection area
        float inter_x_min = std::max(box1.x_min, box2.x_min);
        float inter_y_min = std::max(box1.y_min, box2.y_min);
        float inter_x_max = std::min(box1.x_max, box2.x_max);
        float inter_y_max = std::min(box1.y_max, box2.y_max);
        
        // No intersection
        if (inter_x_max < inter_x_min || inter_y_max < inter_y_min)
            return 0.0f;
        
        // Width and height of intersection area
        float inter_width = inter_x_max - inter_x_min;
        float inter_height = inter_y_max - inter_y_min;
        
        // Area of intersection
        float inter_area = inter_width * inter_height;
        
        // Calculate area of each box
        float box1_area = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min);
        float box2_area = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min);
        
        // IoU = Intersection area / Union area
        float iou = inter_area / (box1_area + box2_area - inter_area);
        
        return iou;
    }

    // Function to perform NMS (Non-Maximum Suppression)
    std::vector<BoundingBox> performNMS(const std::vector<BoundingBox>& boxes) const {
        if (boxes.empty()) return {};
        
        // Sort boxes in descending order by confidence
        std::vector<BoundingBox> sorted_boxes = boxes;
        
        std::vector<BoundingBox> selected_boxes;
        std::vector<bool> is_suppressed(sorted_boxes.size(), false);
        
        for (size_t i = 0; i < sorted_boxes.size(); ++i) {
            if (is_suppressed[i]) continue;
            
            selected_boxes.push_back(sorted_boxes[i]);
            
            // Calculate IoU between current box and remaining boxes, remove duplicates
            for (size_t j = i + 1; j < sorted_boxes.size(); ++j) {
                if (is_suppressed[j]) continue;
                
                // Apply NMS only to boxes of the same class
                if (sorted_boxes[i].label_id == sorted_boxes[j].label_id) {
                    float iou = calculateIoU(sorted_boxes[i], sorted_boxes[j]);
                    if (iou > _iouThreshold) {
                        is_suppressed[j] = true;
                    }
                }
            }
        }
        
        // Select only top K results
        if (selected_boxes.size() > _topK) {
            selected_boxes.resize(_topK);
        }
        
        return selected_boxes;
    }


    // Extract box information from tensor
    std::vector<BoundingBox> extractBoxes() {
        const auto dets_shape = _detsTensor.get_shape();
        const auto labels_shape = _labelsTensor.get_shape();
        
        const float* dets_data = _detsTensor.data<const float>();
        const int64_t* labels_data = _labelsTensor.data<const int64_t>();
        
        // Dimension processing
        size_t batch_size = 1;
        size_t num_detections = 0;
        size_t box_elements = 0;
        
        // Process according to tensor dimensions
        if (dets_shape.size() == 3) {
            // Shape [1, 201, 5]
            batch_size = dets_shape[0];
            num_detections = dets_shape[1];
            box_elements = dets_shape[2];  // 5
        } else if (dets_shape.size() == 2) {
            // Shape [201, 5]
            batch_size = 1;
            num_detections = dets_shape[0];
            box_elements = dets_shape[1];  // 5
        } else {
            //std::cout << "Unsupported tensor shape" << std::endl;
            return {};
        }
        
        std::vector<BoundingBox> boxes;
        
        // Process only for the first image (batch)
        size_t image_id = 0;
        
        // Memory layout analysis
        size_t stride1 = num_detections * box_elements;  // Batch stride
        size_t stride2 = box_elements;                  // Box stride
        
        
        for (size_t i = 0; i < num_detections; ++i) {
            size_t base_idx = image_id * stride1 + i * stride2;
            
            // Confidence value (5th element)
            float confidence = 0.0f;
            if (box_elements >= 5) {
                confidence = dets_data[base_idx + 4];
            } else {
                // If confidence value is not available (rare case)
                // Treat the last element as confidence
                confidence = dets_data[base_idx + box_elements - 1];
            }
            
            // Process only if confidence is above threshold
            if (confidence >= _confidenceThreshold) {
                BoundingBox box;
                // Corrected order: xmin, xmax, ymin, ymax, confidence
                box.x_min = dets_data[base_idx + 0];
                box.y_min = dets_data[base_idx + 1];
                box.x_max = dets_data[base_idx + 2];
                box.y_max = dets_data[base_idx + 3];
                box.confidence = confidence;
                
                // Calculate label index
                size_t label_idx = image_id * num_detections + i;
                if (label_idx < _labelsTensor.get_size()) {
                    box.label_id = labels_data[label_idx];
                } else {
                    // Default label 0 if label information is not available
                    box.label_id = 0;
                }
                
                box.index = i;
                boxes.push_back(box);
            }
        }
        
        return boxes;
    }

public:
    explicit DetectionResult(const ov::Tensor& dets_tensor,
                            const ov::Tensor& labels_tensor,
                            const std::vector<std::string>& image_names = {},
                            float confidence_threshold = 0.5,
                            const std::vector<std::string>& class_names = {},
                            float iou_threshold = 0.45,
                            size_t top_k = 10)  // Output top 10 results (default)
        : _detsTensor(dets_tensor),
          _labelsTensor(labels_tensor),
          _classNames(class_names),
          _imageNames(image_names),
          _confidenceThreshold(confidence_threshold),
          _iouThreshold(iou_threshold),
          _topK(top_k) {
              // Process results immediately in constructor
              process();
          }

    /**
     * @brief Process results (extract bounding boxes and perform NMS)
     */
    void process() {
        // Extract bounding boxes
        std::vector<BoundingBox> boxes = extractBoxes();
        
        // Sort in descending order by confidence
        std::sort(boxes.begin(), boxes.end());
        
        // Select only top K (filter before NMS)
        if (boxes.size() > _topK) {
            boxes.resize(_topK);
        }
        
        // Perform NMS
        _filtered_boxes = performNMS(boxes);
    }
    
    /**
     * @brief Get final bounding box results
     * @return Vector of bounding boxes after filtering and NMS
     */
    const std::vector<BoundingBox>& getFilteredBoxes() const {
        return _filtered_boxes;
    }
    
    /**
     * @brief Get class names
     * @return Vector of class names
     */
    const std::vector<std::string>& getClassNames() const {
        return _classNames;
    }
}; 