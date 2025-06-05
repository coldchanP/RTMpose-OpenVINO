#ifndef _RTM_POSE_TRACKER_OPENVINO_H_
#define _RTM_POSE_TRACKER_OPENVINO_H_

#include "rtmdet_openvino.h"
#include "rtmpose_openvino.h"  // PoseMode is defined here

#include <vector>
#include <memory>

// PoseMode is defined in rtmpose_openvino.h

class RTMPoseTrackerOpenvino
{
public:
    RTMPoseTrackerOpenvino();
    virtual ~RTMPoseTrackerOpenvino();

public:
    bool LoadModel(
        const std::string& det_xml_path,
        const std::string& det_bin_path,
        const std::string& pose_xml_path,
        const std::string& pose_bin_path,
        int detect_interval = 10,
        const std::string& device_name = "CPU"
    );

    // Return multiple DetectBox objects and PosePoint results for each box
    std::pair<std::vector<DetectBox>, std::vector<std::vector<PosePoint>>> Inference(const cv::Mat& input_mat);

    // Function to set the mode
    void SetMode(PoseMode mode);

    // Return the current mode
    PoseMode GetMode() const;

    // Get joint connections
    const std::vector<std::pair<int, int>>& GetJointLinks() const;

private:
    std::unique_ptr<RTMDetOpenvino> m_ptr_rtm_det;
    std::unique_ptr<RTMPoseOpenvino> m_ptr_rtm_pose;
    unsigned int m_frame_num;
    int m_detect_interval;

    // Current pose mode
    PoseMode m_pose_mode;

    // Member variable to store multiple DetectBox objects
    std::vector<DetectBox> m_detect_boxes;

    // Keypoint connection definitions
    std::vector<std::pair<int, int>> m_hand_joint_links;
    std::vector<std::pair<int, int>> m_body_joint_links;
    std::vector<std::pair<int, int>> m_empty_joint_links;
};

#endif // !_RTM_POSE_TRACKER_OPENVINO_H_
