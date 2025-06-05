#include "rtmdet_openvino.h"
#include <chrono>
#include <algorithm>

RTMDetOpenvino::RTMDetOpenvino()
    : m_confidence_threshold(0.4f),
    m_iou_threshold(0.45f),
    m_top_k(10)  // Change to keep only top 5
{

}

RTMDetOpenvino::~RTMDetOpenvino()
{

}

std::vector<DetectBox> RTMDetOpenvino::Inference(const cv::Mat& input_mat)
{
    std::vector<DetectBox> result;

    try {
        // Start measuring preprocessing time
        auto preprocess_start = std::chrono::high_resolution_clock::now();

        // Optimize input image preprocessing
        cv::Mat blob;
        cv::Mat rgb_image;

        // Preallocate target buffer when converting from BGR to RGB for optimization
        rgb_image.create(input_mat.size(), input_mat.type());
        cv::cvtColor(input_mat, rgb_image, cv::COLOR_BGR2RGB);

        // Create blob - blobFromImage is internally optimized
        blob = cv::dnn::blobFromImage(rgb_image, 1.0 / 255.0);

        // End measuring preprocessing time
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> preprocess_time = preprocess_end - preprocess_start;

        // Create tensor matching model input shape (NCHW: 1,3,640,640)
        ov::Tensor input_tensor = ov::Tensor(ov::element::f32,
            { 1, 3, 640, 640 },
            blob.data);

        // Set input tensor and run inference
        m_infer_request.set_input_tensor(input_tensor);

        // Start measuring inference time
        auto infer_start = std::chrono::high_resolution_clock::now();

        // Perform actual inference
        m_infer_request.infer();

        // End measuring inference time
        auto infer_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> infer_time = infer_end - infer_start;

        // Start measuring postprocessing time
        auto postprocess_start = std::chrono::high_resolution_clock::now();

        // Retrieve output tensor - first try index-based access for speed improvement
        ov::Tensor dets_tensor, labels_tensor;

        // Attempt to retrieve output tensor using two methods
        try {
            // First attempt to retrieve tensor by index (fastest method)
            dets_tensor = m_infer_request.get_output_tensor(0);  // dets
            labels_tensor = m_infer_request.get_output_tensor(1);  // labels

            // Check if the shape matches the expected pattern
            auto shape_dets = dets_tensor.get_shape();

            if (shape_dets.size() != 3 && shape_dets.size() != 2) {
                throw std::runtime_error("Unexpected tensor shape");
            }
        }
        catch (const std::exception& e) {
            // Try by name
            try {
                const auto& output_info = m_compiled_model.outputs();
                for (size_t i = 0; i < output_info.size(); i++) {
                    std::string name = output_info[i].get_any_name();
                    if (name.find("dets") != std::string::npos) {
                        dets_tensor = m_infer_request.get_output_tensor(i);
                    }
                    else if (name.find("labels") != std::string::npos) {
                        labels_tensor = m_infer_request.get_output_tensor(i);
                    }
                }

                if (!dets_tensor || !labels_tensor) {
                    throw std::runtime_error("Cannot identify dets and labels tensors by name");
                }
            }
            catch (const std::exception& e) {
                // Retry by swapping order
                dets_tensor = m_infer_request.get_output_tensor(1);
                labels_tensor = m_infer_request.get_output_tensor(0);
            }
        }

        // Create DetectionResult object and process results - minimize memory allocation
        std::vector<std::string> image_names = { "input_image" };
        std::vector<std::string> class_names = { "hand" };
        DetectionResult detection_result(dets_tensor, labels_tensor, image_names,
            m_confidence_threshold, class_names,
            m_iou_threshold, m_top_k);

        // Get filtered boxes
        const auto& filtered_boxes = detection_result.getFilteredBoxes();

        // Reserve result vector in advance to prevent memory reallocation
        result.reserve(filtered_boxes.size());

        // Convert DetectionResult::BoundingBox to DetectBox
        for (const auto& bbox : filtered_boxes) {
            DetectBox box;
            box.left = static_cast<int>(bbox.x_min);
            box.top = static_cast<int>(bbox.y_min);
            box.right = static_cast<int>(bbox.x_max);
            box.bottom = static_cast<int>(bbox.y_max);
            box.score = bbox.confidence;
            box.label = static_cast<int>(bbox.label_id);

            result.push_back(box);
        }

        // Sort in descending order by score - already sorted by NMS, so can be omitted
        // std::sort(result.begin(), result.end(), BoxCompare);

        // End measuring postprocessing time
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> postprocess_time = postprocess_end - postprocess_start;

        // Calculate and output total time
        std::chrono::duration<double, std::milli> total_time = postprocess_end - preprocess_start;
        std::cout << "Detection process: Preprocessing=" << preprocess_time.count()
            << "ms, Inference=" << infer_time.count()
            << "ms, Postprocessing=" << postprocess_time.count()
            << "ms, Total=" << total_time.count() << "ms" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error occurred during detection inference: " << e.what() << std::endl;
    }

    return result;
}
