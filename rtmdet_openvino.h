#ifndef _RTM_DET_OPENVINO_H_
#define _RTM_DET_OPENVINO_H_

#include <string>
#include <vector>

#include "openvino_model_base.h"
#include "rtmpose_utils.h"
#include "detection_results.h"

class RTMDetOpenvino : public OpenvinoModelBase
{
public:
    RTMDetOpenvino();
    virtual ~RTMDetOpenvino();

public:
    std::vector<DetectBox> Inference(const cv::Mat& input_mat);

private:
    float m_confidence_threshold;
    float m_iou_threshold;
    size_t m_top_k;
};

#endif // !_RTM_DET_OPENVINO_H_ 