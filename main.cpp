#include <iostream>
#include "opencv2/opencv.hpp"

#include "rtmpose_utils.h"
#include "rtmpose_tracker_openvino.h"

int main()
{
    cv::setNumThreads(4);
    cv::setUseOptimized(true);

    std::string rtm_detnano_xml_path = "./model/face_det.xml";
    std::string rtm_detnano_bin_path = "./model/face_det.bin";

    std::string rtm_pose_xml_path = "./model/face_pose.xml";
    std::string rtm_pose_bin_path = "./model/face_pose.bin";

    RTMPoseTrackerOpenvino rtmpose_tracker_openvino;


    // 0: BODY, 1: FACE, 2: HAND
    int mode = 1;
    float pose_thres = 0.2;

    switch (mode) {
    case 0:
        rtmpose_tracker_openvino.SetMode(PoseMode::BODY);
        std::cout << "Mode : Body" << std::endl;
        break;

    case 1:
        rtmpose_tracker_openvino.SetMode(PoseMode::FACE);
        std::cout << "Mode : Face" << std::endl;
        break;
    case 2:
    default:
        rtmpose_tracker_openvino.SetMode(PoseMode::HAND);
        std::cout << "Mode : Hand" << std::endl;
        break;
    }

    // Load models
    std::cout << "Loading models..." << std::endl;
    std::cout << "Detection Model XML: " << rtm_detnano_xml_path << std::endl;
    std::cout << "Detection Model BIN: " << rtm_detnano_bin_path << std::endl;
    std::cout << "Pose Model XML: " << rtm_pose_xml_path << std::endl;
    std::cout << "Pose Model BIN: " << rtm_pose_bin_path << std::endl;

    bool load_model_result = rtmpose_tracker_openvino.LoadModel(
        rtm_detnano_xml_path,
        rtm_detnano_bin_path,
        rtm_pose_xml_path,
        rtm_pose_bin_path,
        2,  // Set detect_interval to 3 for more frequent detection
        "CPU" // Additional option: "CPU", "GPU", "AUTO"
    );

    if (!load_model_result)
    {
        std::cout << "Failed to load OpenVINO model!" << std::endl;
        return 0;
    }

    // Video file path
    std::string video_path = "./vid/test.mp4";
    cv::VideoCapture video_reader(0);

    if (!video_reader.isOpened()) {
        std::cout << "Video file error: " << video_path << std::endl;
        return 0;
    }

    // Get video properties
    int frame_width = static_cast<int>(video_reader.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(video_reader.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = video_reader.get(cv::CAP_PROP_FPS);

    std::cout << "Video info: " << frame_width << "x" << frame_height
        << ", FPS: " << fps << std::endl;

    // Skip unnecessary decoding - optimize buffering
    video_reader.set(cv::CAP_PROP_BUFFERSIZE, 3);

    // Pre-allocate memory to avoid reallocation
    cv::Mat frame, frame_resize;
    frame_resize = cv::Mat(cv::Size(640, 640), CV_8UC3);

    // Create and configure window
    cv::namedWindow("RTMPose OpenVINO", cv::WINDOW_NORMAL);
    cv::resizeWindow("RTMPose OpenVINO", 800, 600);

    int frame_num = 0;

    // Performance measurement variables
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time, current_time;
    double avg_fps = 0.0;
    int fps_counter = 0;
    const int FPS_WINDOW = 30; // Calculate average FPS over 30 frames

    last_time = std::chrono::high_resolution_clock::now();

    while (video_reader.isOpened())
    {
        // Read frame
        bool success = video_reader.read(frame);
        if (!success || frame.empty())
            break;

        // Record current time
        current_time = std::chrono::high_resolution_clock::now();

        // Preprocess input image
        float scale = LetterBoxImage(frame, frame_resize, cv::Size(640, 640), 32, cv::Scalar(128, 128, 128), true);

        // Inference start time
        auto inference_start = std::chrono::high_resolution_clock::now();

        // Retrieve detection boxes and pose results at once
        auto inference_result = rtmpose_tracker_openvino.Inference(frame_resize);
        std::vector<DetectBox> detect_boxes = inference_result.first;
        std::vector<std::vector<PosePoint>> all_pose_results = inference_result.second;

        // Inference end time
        auto inference_end = std::chrono::high_resolution_clock::now();

        // Get connection line information based on the current mode
        const auto& joint_links = rtmpose_tracker_openvino.GetJointLinks();

        // Visualization start time
        auto visualize_start = std::chrono::high_resolution_clock::now();

        // If detect_boxes is empty, nothing is drawn.
        if (!detect_boxes.empty())
        {
            for (size_t i = 0; i < detect_boxes.size(); ++i)
            {
                // Check if pose result exists for this detection box
                if (i >= all_pose_results.size()) {
                    continue; // Skip if no corresponding pose result
                }

                DetectBox detect_box = detect_boxes[i];
                detect_box.left = static_cast<int>(detect_box.left * scale);
                detect_box.right = static_cast<int>(detect_box.right * scale);
                detect_box.top = static_cast<int>(detect_box.top * scale);
                detect_box.bottom = static_cast<int>(detect_box.bottom * scale);

                std::vector<PosePoint> pose_result = all_pose_results[i];

                if (detect_box.IsValid())
                {
                    // Draw bounding box
                    cv::rectangle(
                        frame,
                        cv::Point(detect_box.left, detect_box.top),
                        cv::Point(detect_box.right, detect_box.bottom),
                        cv::Scalar{ 255, 0, 0 },
                        1);

                    // Only draw connection lines if the current mode is not FACE and connection lines are defined
                    if (rtmpose_tracker_openvino.GetMode() != PoseMode::FACE && !joint_links.empty())
                    {
                        // Draw keypoint connection lines
                        for (const auto& link : joint_links)
                        {
                            int idx1 = link.first;
                            int idx2 = link.second;

                            // Check array bounds and score threshold
                            if (idx1 < pose_result.size() && idx2 < pose_result.size() &&
                                pose_result[idx1].score > pose_thres && pose_result[idx2].score > pose_thres)
                            {
                                cv::line(
                                    frame,
                                    cv::Point(static_cast<int>(pose_result[idx1].x * scale), static_cast<int>(pose_result[idx1].y * scale)),
                                    cv::Point(static_cast<int>(pose_result[idx2].x * scale), static_cast<int>(pose_result[idx2].y * scale)),
                                    cv::Scalar{ 0, 255, 0 },
                                    1,
                                    cv::LINE_AA);
                            }
                        }
                    }

                    for (const auto& point : pose_result)
                    {
                        if (point.score > pose_thres) {
                            cv::Scalar color(0, 255, 255);
                            cv::circle(frame,
                                cv::Point(static_cast<int>(point.x * scale), static_cast<int>(point.y * scale)),
                                2,
                                color,
                                -1,
                                cv::LINE_AA);
                        }
                    }
                }
            }
        }

        // Calculate and display FPS
        std::chrono::duration<double> frame_time = current_time - last_time;
        double current_fps = 1.0 / frame_time.count();

        // Calculate moving average FPS
        avg_fps = (avg_fps * fps_counter + current_fps) / (fps_counter + 1);
        if (++fps_counter >= FPS_WINDOW) {
            fps_counter = 0;
        }

        cv::putText(frame,
            "FPS: " + std::to_string(static_cast<int>(avg_fps)),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2);

        // Visualization end time
        auto visualize_end = std::chrono::high_resolution_clock::now();

        // Timing results
        std::chrono::duration<double, std::milli> inference_time = inference_end - inference_start;
        std::chrono::duration<double, std::milli> visualize_time = visualize_end - visualize_start;
        std::chrono::duration<double, std::milli> total_time = visualize_end - inference_start;

        // Print performance information every 30 frames
        if (frame_num % 30 == 0) {
            std::cout << "Frame " << frame_num
                << " processing time: Inference=" << inference_time.count()
                << "ms, Visualization=" << visualize_time.count()
                << "ms, Total=" << total_time.count()
                << "ms, FPS=" << avg_fps << std::endl;
        }

        // Display frame
        cv::imshow("RTMPose OpenVINO", frame);

        // Update time
        last_time = current_time;
        frame_num++;

        // Exit if ESC key or 'q' key is pressed
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
            break;
    }

    // Release resources
    video_reader.release();
    cv::destroyAllWindows();

    return 0;
}
