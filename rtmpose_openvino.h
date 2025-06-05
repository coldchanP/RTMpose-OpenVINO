#ifndef _RTM_POSE_OPENVINO_H_
#define _RTM_POSE_OPENVINO_H_

#include <string>
#include <vector>

#include "openvino_model_base.h"
#include "rtmpose_utils.h"
#include "pose_results.h"

// Define pose estimation modes
enum class PoseMode {
    BODY = 0,
    FACE = 1,
    HAND = 2
};

class RTMPoseOpenvino : public OpenvinoModelBase
{
public:
    RTMPoseOpenvino();
    virtual ~RTMPoseOpenvino();

public:
    std::vector<std::vector<PosePoint>> Inference(const cv::Mat& input_mat, const std::vector<DetectBox>& boxes);

    // Function to set the mode
    void SetMode(PoseMode mode);

    // Return the current mode
    PoseMode GetMode() const;

private:
    // Add output size parameters to the CropImageByDetectBox method
    std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat& input_image, const DetectBox& box,
        int output_width = 256, int output_height = 256);
    float m_confidence_threshold;
    PoseMode m_pose_mode;  // Store the current pose mode
};

#endif // !_RTM_POSE_OPENVINO_H_
