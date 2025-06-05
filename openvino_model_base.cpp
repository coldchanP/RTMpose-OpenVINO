// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino_model_base.h"
#include <iostream>

OpenvinoModelBase::OpenvinoModelBase()
{
}

OpenvinoModelBase::~OpenvinoModelBase()
{
}

bool OpenvinoModelBase::LoadModel(const std::string& xml_model_path, const std::string& bin_model_path, const std::string& device_name)
{
    try
    {
        // Load model
        std::shared_ptr<ov::Model> model;

        if (bin_model_path.empty()) {
            model = m_core.read_model(xml_model_path);
        }
        else {
            model = m_core.read_model(xml_model_path, bin_model_path);
        }

        // Compile model and create inference request
        m_compiled_model = m_core.compile_model(model, device_name);
        m_infer_request = m_compiled_model.create_infer_request();

        // Print model information
        PrintModelInfo(m_compiled_model);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "OpenVINO model loading failed: " << e.what() << std::endl;
        return false;
    }
}

void OpenvinoModelBase::PrintModelInfo(const ov::CompiledModel& compiled_model)
{
    try
    {
        // Input information
        std::cout << "Model input information:" << std::endl;
        for (const auto& input : compiled_model.inputs()) {
            std::cout << "Input name: " << input.get_any_name() << std::endl;
            std::cout << "Input shape: " << input.get_shape() << std::endl;
            std::cout << "Input type: " << input.get_element_type() << std::endl;
            std::cout << std::endl;
        }

        // Output information
        std::cout << "Model output information:" << std::endl;
        for (const auto& output : compiled_model.outputs()) {
            std::cout << "Output name: " << output.get_any_name() << std::endl;
            std::cout << "Output shape: " << output.get_shape() << std::endl;
            std::cout << "Output type: " << output.get_element_type() << std::endl;
            std::cout << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to print model info: " << e.what() << std::endl;
    }
}
