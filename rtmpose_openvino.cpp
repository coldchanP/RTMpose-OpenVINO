#include "rtmpose_openvino.h"
#include <chrono>
#include <algorithm>

RTMPoseOpenvino::RTMPoseOpenvino()
    : m_confidence_threshold(0.05f),
    m_pose_mode(PoseMode::HAND) // Default is HAND
{
}

RTMPoseOpenvino::~RTMPoseOpenvino()
{
}

void RTMPoseOpenvino::SetMode(PoseMode mode)
{
    m_pose_mode = mode;
}

PoseMode RTMPoseOpenvino::GetMode() const
{
    return m_pose_mode;
}

std::vector<std::vector<PosePoint>> RTMPoseOpenvino::Inference(const cv::Mat& input_mat, const std::vector<DetectBox>& boxes)
{
    std::vector<std::vector<PosePoint>> all_pose_results;

    // Quickly return if the box list is empty
    if (boxes.empty()) {
        return all_pose_results;
    }

    // Reserve space for results to avoid reallocation
    all_pose_results.reserve(boxes.size());

    try {
        // Determine input size based on mode
        //int input_width = (m_pose_mode == PoseMode::BODY) ? 320 : 256;
        //int input_height = (m_pose_mode == PoseMode::BODY) ? 320 : 256;
    	int input_width = 256;
        int input_height = 256;

        // Preallocate memory for output images
        cv::Mat resized(cv::Size(input_width, input_height), CV_8UC3);
        cv::Mat rgb_image(cv::Size(input_width, input_height), CV_8UC3);

        // Start measuring total processing time
        auto total_start = std::chrono::high_resolution_clock::now();

        // Total time for all cropping and transformation operations
        double total_crop_time = 0.0;
        // Total time for all inference operations
        double total_infer_time = 0.0;
        // Total time for all postprocessing operations
        double total_postprocess_time = 0.0;

        for (const auto& box : boxes) {
            if (!box.IsValid()) {
                // Add empty pose result for invalid box to maintain index consistency
                all_pose_results.push_back(std::vector<PosePoint>());
                continue;
            }

            try {
                // Start measuring crop time
                auto crop_start = std::chrono::high_resolution_clock::now();

                // Crop the image region from the detected bounding box
                auto cropped_result = CropImageByDetectBox(input_mat, box, input_width, input_height);
                cv::Mat& cropped_image = cropped_result.first;
                cv::Mat& affine_transform_matrix = cropped_result.second;

                // Image preprocessing (resize, normalize) - using preallocated memory
                cv::resize(cropped_image, resized, cv::Size(input_width, input_height));
                cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);

                cv::Mat blob;
                blob = cv::dnn::blobFromImage(rgb_image, 1.0 / 255.0);

                // End measuring crop time
                auto crop_end = std::chrono::high_resolution_clock::now();
                total_crop_time += std::chrono::duration<double, std::milli>(crop_end - crop_start).count();

                // Create tensor matching model input shape (NCHW: 1,3,input_width,input_height)
                ov::Tensor input_tensor = ov::Tensor(ov::element::f32,
                    { 1, 3, static_cast<size_t>(input_height), static_cast<size_t>(input_width) },
                    blob.data);

                // Set input tensor and run inference
                m_infer_request.set_input_tensor(input_tensor);

                // Start measuring inference time
                auto infer_start = std::chrono::high_resolution_clock::now();
                m_infer_request.infer();
                auto infer_end = std::chrono::high_resolution_clock::now();
                total_infer_time += std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

                // Start measuring postprocessing time
                auto postprocess_start = std::chrono::high_resolution_clock::now();

                // Retrieve output tensors using direct index access
                ov::Tensor output_tensor_x, output_tensor_y;

                // Attempt to retrieve output tensors using two methods - optimized approach
            	// First, try retrieving tensors by index (fastest)
            	output_tensor_x = m_infer_request.get_output_tensor(0);  // simcc_x
            	output_tensor_y = m_infer_request.get_output_tensor(1);  // simcc_y


                if (!output_tensor_x || !output_tensor_y) {
                    throw std::runtime_error("Unable to retrieve output tensors");
                }

                // Create PoseResult object
                PoseResult pose_result(output_tensor_x, output_tensor_y, {}, m_confidence_threshold, {}, 2.0f,
                    static_cast<float>(input_width), static_cast<float>(input_height));

                // Get keypoints
                const auto& keypoints = pose_result.getFilteredKeypoints();

                // Reserve vector based on keypoint count to avoid reallocation
                std::vector<PosePoint> pose_points;
                pose_points.reserve(keypoints.size());

                // Convert PoseResult::Keypoint to PosePoint
                for (const auto& kp : keypoints) {
                    PosePoint point;

                    // Transform keypoint to original image coordinate system - optimized by direct computation
                    cv::Point2f orig_pt(static_cast<float>(kp.x), static_cast<float>(kp.y));
                    cv::Point2f transformed_pt;

                    // Apply inverse transformation (using the inverse of the affine transform matrix) - direct access for performance improvement
                    cv::Mat inv_affine_transform;
                    cv::invertAffineTransform(affine_transform_matrix, inv_affine_transform);

                    // Apply transformation - direct computation
                    transformed_pt.x = inv_affine_transform.at<double>(0, 0) * orig_pt.x +
                        inv_affine_transform.at<double>(0, 1) * orig_pt.y +
                        inv_affine_transform.at<double>(0, 2);
                    transformed_pt.y = inv_affine_transform.at<double>(1, 0) * orig_pt.x +
                        inv_affine_transform.at<double>(1, 1) * orig_pt.y +
                        inv_affine_transform.at<double>(1, 2);

                    point.x = static_cast<int>(transformed_pt.x);
                    point.y = static_cast<int>(transformed_pt.y);
                    point.score = kp.confidence;

                    pose_points.push_back(point);
                }

                all_pose_results.push_back(pose_points);

                // End measuring postprocessing time
                auto postprocess_end = std::chrono::high_resolution_clock::now();
                total_postprocess_time += std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing box: " << e.what() << " - Adding empty pose result" << std::endl;
                // Add empty pose result for this box to maintain index consistency
                all_pose_results.push_back(std::vector<PosePoint>());
            }
        }

        // End total processing time measurement
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Calculate average times
        double avg_crop_time = boxes.empty() ? 0 : total_crop_time / boxes.size();
        double avg_infer_time = boxes.empty() ? 0 : total_infer_time / boxes.size();
        double avg_postprocess_time = boxes.empty() ? 0 : total_postprocess_time / boxes.size();

        std::cout << "Average pose time per box: Preprocessing=" << avg_crop_time
            << "ms, Inference=" << avg_infer_time
            << "ms, Postprocessing=" << avg_postprocess_time
            << "ms, Total=" << total_time << "ms" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error occurred during pose inference: " << e.what() << std::endl;
    }

    return all_pose_results;
}

std::pair<cv::Mat, cv::Mat> RTMPoseOpenvino::CropImageByDetectBox(const cv::Mat& input_image, const DetectBox& box, int output_width, int output_height)
{
    std::pair<cv::Mat, cv::Mat> result_pair;

    // Get image dimensions
    int img_width = input_image.cols;
    int img_height = input_image.rows;

    // Ensure box coordinates are valid and within image bounds
    int left = std::max(0, std::min(box.left, img_width - 2));
    int top = std::max(0, std::min(box.top, img_height - 2));
    int right = std::max(left + 2, std::min(box.right, img_width - 1));
    int bottom = std::max(top + 2, std::min(box.bottom, img_height - 1));

    // Calculate box center
    float box_center_x = (left + right) / 2.0f;
    float box_center_y = (top + bottom) / 2.0f;

    // Calculate box size (after boundary adjustment)
    float box_width = right - left;
    float box_height = bottom - top;

    // Ensure minimum box dimensions
    box_width = std::max(box_width, 10.0f);
    box_height = std::max(box_height, 10.0f);

    // Scale the box size by 1.2x
    float scale_image_width = box_width * 1.2f;
    float scale_image_height = box_height * 1.2f;

    // Calculate safe source rectangle that doesn't exceed image boundaries
    float safe_left = std::max(0.0f, box_center_x - scale_image_width / 2.0f);
    float safe_top = std::max(0.0f, box_center_y - scale_image_height / 2.0f);
    float safe_right = std::min(static_cast<float>(img_width - 1), box_center_x + scale_image_width / 2.0f);
    float safe_bottom = std::min(static_cast<float>(img_height - 1), box_center_y + scale_image_height / 2.0f);

    // Recalculate dimensions to maintain aspect ratio close to the original scale
    float actual_width = safe_right - safe_left;
    float actual_height = safe_bottom - safe_top;

    // Safeguard against zero-sized regions
    if (actual_width < 4.0f || actual_height < 4.0f) {
        // Create a minimal valid region in the center of the image if the original region is too small
        safe_left = img_width / 2.0f - 50.0f;
        safe_top = img_height / 2.0f - 50.0f;
        safe_right = img_width / 2.0f + 50.0f;
        safe_bottom = img_height / 2.0f + 50.0f;
        
        // Re-adjust to stay within image boundaries
        safe_left = std::max(0.0f, safe_left);
        safe_top = std::max(0.0f, safe_top);
        safe_right = std::min(static_cast<float>(img_width - 1), safe_right);
        safe_bottom = std::min(static_cast<float>(img_height - 1), safe_bottom);
        
        actual_width = safe_right - safe_left;
        actual_height = safe_bottom - safe_top;
    }

    // Create source points for affine transformation
    std::vector<cv::Point2f> src_points = {
        cv::Point2f(safe_left, safe_top),                // Top-left
        cv::Point2f(safe_right, safe_top),               // Top-right
        cv::Point2f(safe_left, safe_bottom)              // Bottom-left
    };

    // Destination points in the output image
    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(256.0f, 0),
        cv::Point2f(0, 256.0f)
    };

    // Calculate affine transformation matrix
    cv::Mat affine_transform = cv::getAffineTransform(src_points, dst_points);

    // Apply affine transformation to crop and resize the image
    cv::Mat affine_image;
    try {
        cv::warpAffine(input_image, affine_image, affine_transform, cv::Size(output_width, output_height), cv::INTER_LINEAR);
    }
    catch (const cv::Exception& e) {
        // Fallback in case of transformation failure
        std::cerr << "Affine transformation error: " << e.what() << std::endl;
        
        // Create a blank image as fallback
        affine_image = cv::Mat(output_width, output_height, input_image.type(), cv::Scalar(0, 0, 0));
        
        // Create identity transformation as fallback
        affine_transform = cv::Mat::eye(2, 3, CV_64F);
    }

    result_pair = std::make_pair(affine_image, affine_transform);
    return result_pair;
}
