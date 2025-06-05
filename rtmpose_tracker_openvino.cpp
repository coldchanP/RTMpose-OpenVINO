#include "rtmpose_tracker_openvino.h"
#include <chrono>
#include <algorithm>

RTMPoseTrackerOpenvino::RTMPoseTrackerOpenvino()
    :m_ptr_rtm_det(nullptr),
    m_ptr_rtm_pose(nullptr),
    m_frame_num(0),
    m_detect_interval(5),  // Detection interval set to 5 frames
    m_pose_mode(PoseMode::HAND)  // Default mode is HAND
{
    // Initialize hand keypoint connections
    m_hand_joint_links = {
        {0,1},{1,2},{2,3},{3,4},{0,5},{0,9},{0,13},{0,17},{5,6},{6,7},{7,8},{9,10},
        {10,11},{11,12},{13,14},{14,15},{15,16},{17,18},{18,19},{19,20}
    };

    // Initialize body keypoint connections
    m_body_joint_links = {
        {0,1},{0,2},{0,17},{0,18},{18,5},{2,4},{1,3},{6,18},{6,8},{8,10},{5,7},{7,9},
        {18,19},{19,12},{19,11},{12,14},{14,16},{16,21},{16,23},{16,25},{11,13},{13,15},
        {15,20},{15,22},{15,24}
    };

    // Empty connection list for face
    m_empty_joint_links.clear();
}

RTMPoseTrackerOpenvino::~RTMPoseTrackerOpenvino()
{
}

void RTMPoseTrackerOpenvino::SetMode(PoseMode mode)
{
    m_pose_mode = mode;

    // If pose instance has already been created, set the pose mode as well
    if (m_ptr_rtm_pose) {
        m_ptr_rtm_pose->SetMode(mode);
    }
}

PoseMode RTMPoseTrackerOpenvino::GetMode() const
{
    return m_pose_mode;
}

const std::vector<std::pair<int, int>>& RTMPoseTrackerOpenvino::GetJointLinks() const
{
    switch (m_pose_mode) {
    case PoseMode::BODY:
        return m_body_joint_links;
    case PoseMode::HAND:
        return m_hand_joint_links;
    case PoseMode::FACE:
    default:
        return m_empty_joint_links;
    }
}

bool RTMPoseTrackerOpenvino::LoadModel(
    const std::string& det_xml_path,
    const std::string& det_bin_path,
    const std::string& pose_xml_path,
    const std::string& pose_bin_path,
    int detect_interval,
    const std::string& device_name)
{
    // Initialize detection model
    m_ptr_rtm_det = std::make_unique<RTMDetOpenvino>();
    if (m_ptr_rtm_det == nullptr)
        return false;

    if (!m_ptr_rtm_det->LoadModel(det_xml_path, det_bin_path, device_name))
        return false;

    // Initialize pose estimation model
    m_ptr_rtm_pose = std::make_unique<RTMPoseOpenvino>();
    if (m_ptr_rtm_pose == nullptr)
        return false;

    // Set the pose mode before loading the pose model
    m_ptr_rtm_pose->SetMode(m_pose_mode);

    if (!m_ptr_rtm_pose->LoadModel(pose_xml_path, pose_bin_path, device_name))
        return false;

    m_detect_interval = detect_interval;

    return true;
}

std::pair<std::vector<DetectBox>, std::vector<std::vector<PosePoint>>> RTMPoseTrackerOpenvino::Inference(const cv::Mat& input_mat)
{
    std::pair<std::vector<DetectBox>, std::vector<std::vector<PosePoint>>> result;

    if (m_ptr_rtm_det == nullptr || m_ptr_rtm_pose == nullptr)
        return result;

    // Use high_resolution_clock for better precision in performance measurement
    auto detect_start = std::chrono::high_resolution_clock::now();
    auto detect_end = detect_start;

    // Periodic detection based on detection interval
    if (m_frame_num % m_detect_interval == 0)
    {
        // Start measuring detection time
        detect_start = std::chrono::high_resolution_clock::now();

        // Add parallel processing hint (use if supported by OpenCV)
        cv::setNumThreads(4); // Adjust according to available cores

        m_detect_boxes = m_ptr_rtm_det->Inference(input_mat);  // Return multiple DetectBoxes

        // End detection time measurement and output the elapsed time
        detect_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detect_time = detect_end - detect_start;

        // If there are many results, keep only the top N to reduce pose inference time
        if (m_detect_boxes.size() > 10) {
            // Since the results are already sorted by score, keep only the top 5
            m_detect_boxes.resize(10);
        }

        std::cout << "Detection inference time: " << detect_time.count() << " milliseconds" << std::endl;
    }
    else if (m_detect_boxes.empty()) {
        // If no boxes were previously detected, force detection
        detect_start = std::chrono::high_resolution_clock::now();
        m_detect_boxes = m_ptr_rtm_det->Inference(input_mat);
        detect_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detect_time = detect_end - detect_start;
        std::cout << "Forced detection inference time: " << detect_time.count() << " milliseconds" << std::endl;
    }

    // Start measuring pose inference time
    auto pose_start = std::chrono::high_resolution_clock::now();

    // If pose mode was changed, pass the mode to RTMPoseOpenvino as well
    m_ptr_rtm_pose->SetMode(m_pose_mode);

    // Perform pose inference for each box
    std::vector<std::vector<PosePoint>> all_pose_results = m_ptr_rtm_pose->Inference(input_mat, m_detect_boxes);

    // End measuring pose inference time
    auto pose_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pose_time = pose_end - pose_start;

    // Calculate total inference time
    std::chrono::duration<double, std::milli> total_time = pose_end - detect_start;

    double fps = 1000.0 / total_time.count();

    std::cout << "Pose inference time: " << pose_time.count() << " milliseconds, total processing time: "
        << total_time.count() << " milliseconds, FPS: " << fps << std::endl;

    m_frame_num += 1;

    // Return a pair of multiple DetectBoxes and PosePoint lists
    return std::make_pair(m_detect_boxes, all_pose_results);
}
