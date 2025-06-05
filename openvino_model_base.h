// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_MODEL_BASE_H_
#define _OPENVINO_MODEL_BASE_H_

#include <string>
#include <thread>

#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"

class OpenvinoModelBase
{
public:
    OpenvinoModelBase();
    virtual ~OpenvinoModelBase();

public:
    virtual bool LoadModel(const std::string& xml_model_path, const std::string& bin_model_path = "", const std::string& device_name = "CPU");

protected:
    virtual void PrintModelInfo(const ov::CompiledModel& compiled_model);

protected:
    ov::Core m_core;
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_infer_request;
};

#endif // !_OPENVINO_MODEL_BASE_H_ 