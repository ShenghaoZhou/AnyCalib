#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

/**
 * AnyCalib C++ TensorRT Inference
 * 
 * This code implements the preprocessing, model execution, and post-processing 
 * (linear fit) required to estimate camera intrinsics from an image.
 */

// Simple Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

struct CameraParams {
    float fx, fy, cx, cy;
};

class AnyCalibTRT {
public:
    AnyCalibTRT(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) throw std::runtime_error("Could not read engine file: " + engine_path);

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        char* data = new char[size];
        file.read(data, size);
        file.close();

        runtime = nvinfer1::createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(data, size);
        context = engine->createExecutionContext();
        delete[] data;

        // Automatically detect buffer sizes
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            auto name = engine->getIOTensorName(i);
            
            // For dynamic shapes, we need to allocate for the maximum possible size
            auto max_dims = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            size_t vol = 1;
            for (int j = 0; j < max_dims.nbDims; ++j) {
                if (max_dims.d[j] > 0) vol *= max_dims.d[j];
            }
            if (vol < 3 * 1024 * 1024) vol = 3 * 1024 * 1024; // Force at least 12MB per tensor
            
            void* device_ptr;
            cudaMalloc(&device_ptr, vol * sizeof(float));
            buffers.push_back(device_ptr);
            tensor_names.push_back(name);
            tensor_sizes.push_back(vol);
        }
    }

    ~AnyCalibTRT() {
        for (void* buf : buffers) cudaFree(buf);
        delete context;
        delete engine;
        delete runtime;
    }

    CameraParams run(const cv::Mat& input_image) {
        int ho = input_image.rows;
        int wo = input_image.cols;

        // 1. Calculate target size based on 102,400 pixel target (approx 322x322)
        // DINOv2 requires dimensions to be multiples of 14.
        float target_res = 102400.0f;
        float ar = std::max(0.5f, std::min((float)ho / wo, 2.0f));
        float w_ideal = std::sqrt(target_res / ar);
        float h_ideal = ar * w_ideal;
        int wt = std::round(w_ideal / 14) * 14;
        int ht = std::round(h_ideal / 14) * 14;

        // 2. Preprocess: Center Crop and Normalize
        cv::Mat processed, scale_xy, shift_xy;
        preprocess(input_image, processed, ht, wt, scale_xy, shift_xy);

        // 3. TensorRT Inference
        // Set dynamic input shape
        nvinfer1::Dims4 input_dims{1, 3, ht, wt};
        context->setInputShape("image", input_dims);

        int image_idx = -1;
        for (int i = 0; i < tensor_names.size(); ++i) {
            if (tensor_names[i] == "image") {
                image_idx = i;
                break;
            }
        }
        cudaMemcpy(buffers[image_idx], processed.data, (ht * wt * 3) * sizeof(float), cudaMemcpyHostToDevice);
        for (int i = 0; i < tensor_names.size(); ++i) {
            context->setTensorAddress(tensor_names[i].c_str(), buffers[i]);
        }
        bool status = context->enqueueV3(0);
        if (!status) {
            std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        }
        cudaStreamSynchronize(0);

        int rays_idx = -1;
        for (int i = 0; i < tensor_names.size(); ++i) {
            if (tensor_names[i] == "rays") {
                rays_idx = i;
                break;
            }
        }
        
        auto output_dims = context->getTensorShape(tensor_names[rays_idx].c_str());
        
        size_t output_vol = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) output_vol *= output_dims.d[i];
        
        std::vector<float> h_rays(output_vol);
        cudaMemcpy(h_rays.data(), buffers[rays_idx], output_vol * sizeof(float), cudaMemcpyDeviceToHost);

        // 4. Post-process: Linear Least Squares for Pinhole parameters
        CameraParams pred = linear_fit(h_rays, ht, wt);

        // 5. De-normalization of parameters (undo the initial crop/resize)
        // Formula: f_orig = f_pred / scale,  c_orig = (c_pred - shift) / scale
        pred.fx /= scale_xy.at<float>(0);
        pred.fy /= scale_xy.at<float>(1);
        pred.cx = (pred.cx - shift_xy.at<float>(0)) / scale_xy.at<float>(0);
        pred.cy = (pred.cy - shift_xy.at<float>(1)) / scale_xy.at<float>(1);

        return pred;
    }

private:
    void preprocess(const cv::Mat& src, cv::Mat& dst, int ht, int wt, cv::Mat& scale_xy, cv::Mat& shift_xy) {
        cv::Mat img;
        cv::cvtColor(src, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);

        int h = img.rows, w = img.cols;
        float s1 = 1.0f;
        cv::Vec2f s1_xy(1.0f, 1.0f);
        
        // Upscale if smaller than target
        if (h < ht || w < wt) {
            s1 = std::max((float)ht / h, (float)wt / w);
            cv::resize(img, img, cv::Size(), s1, s1, cv::INTER_CUBIC);
            s1_xy = cv::Vec2f((float)img.cols / w, (float)img.rows / h);
            h = img.rows; w = img.cols;
        }

        // Center search for cropping
        cv::Vec2f shift(0, 0);
        float ar_t = (float)wt / ht;
        if ((float)w / h > ar_t) {
            int crop_w = std::round(w - h * ar_t);
            int start_x = crop_w / 2;
            img = img(cv::Rect(start_x, 0, w - crop_w, h));
            shift[0] = -start_x;
        } else {
            int crop_h = std::round(h - w / ar_t);
            int start_y = crop_h / 2;
            img = img(cv::Rect(0, start_y, w, h - crop_h));
            shift[1] = -start_y;
        }

        // Downsample to final target resolution
        cv::Vec2f s2_xy((float)wt / img.cols, (float)ht / img.rows);
        cv::resize(img, img, cv::Size(wt, ht), 0, 0, cv::INTER_AREA);
        
        scale_xy = cv::Mat(cv::Vec2f(s1_xy[0] * s2_xy[0], s1_xy[1] * s2_xy[1])).clone();
        shift_xy = cv::Mat(cv::Vec2f(shift[0] * s2_xy[0], shift[1] * s2_xy[1])).clone();

        // Standard DINOv2 / ImageNet normalization
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(0.229, 0.224, 0.225);
        cv::subtract(img, mean, img);
        cv::divide(img, std, img);

        // Transform HWC to CHW planar format for the network
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);
        dst = cv::Mat(ht * wt * 3, 1, CV_32F);
        for (int i = 0; i < 3; ++i) {
            memcpy(dst.data + i * ht * wt * sizeof(float), channels[i].data, ht * wt * sizeof(float));
        }
    }

    CameraParams linear_fit(const std::vector<float>& rays, int h, int w) {
        int n = h * w;
        // Solve A*x = B separately for x and y focal/principal points
        // System: u/N = (1/f)*X/Z + (c/f) where f is unknown
        // Reparameterized as: X/Z = a*(u) - b where a=1/f, b=c/f
        cv::Mat Ax(n, 2, CV_32F), Ay(n, 2, CV_32F);
        cv::Mat Bx(n, 1, CV_32F), By(n, 1, CV_32F);

        float N_u = w, N_v = h; // Norm factors for numerical stability

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int idx = i * w + j;
                float rz = std::max(rays[2 * n + idx], 1e-6f);
                float x_z = rays[0 * n + idx] / rz;
                float y_z = rays[1 * n + idx] / rz;

                float u = j + 0.5f; // Pixel centers
                float v = i + 0.5f;

                Ax.at<float>(idx, 0) = u / N_u;
                Ax.at<float>(idx, 1) = -1.0f;
                Bx.at<float>(idx) = x_z;

                Ay.at<float>(idx, 0) = v / N_v;
                Ay.at<float>(idx, 1) = -1.0f;
                By.at<float>(idx) = y_z;
            }
        }

        cv::Mat sol_x, sol_y;
        cv::solve(Ax, Bx, sol_x, cv::DECOMP_SVD);
        cv::solve(Ay, By, sol_y, cv::DECOMP_SVD);

        float fx = N_u / sol_x.at<float>(0);
        float fy = N_v / sol_y.at<float>(0);
        float cx = sol_x.at<float>(1) * fx;
        float cy = sol_y.at<float>(1) * fy;

        return {fx, fy, cx, cy};
    }

    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::vector<void*> buffers;
    std::vector<std::string> tensor_names;
    std::vector<size_t> tensor_sizes;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
        return -1;
    }

    try {
        AnyCalibTRT model(argv[1]);
        cv::Mat img = cv::imread(argv[2]);
        if (img.empty()) {
            std::cerr << "Could not open image: " << argv[2] << std::endl;
            return -1;
        }

        CameraParams res = model.run(img);

        std::cout << "\n=== Predicted Camera Intrinsic Parameters ===" << std::endl;
        std::cout << "Image Resolution: " << img.cols << "x" << img.rows << std::endl;
        std::cout << "Focal Length $fx: " << res.fx << std::endl;
        std::cout << "Focal Length $fy: " << res.fy << std::endl;
        std::cout << "Principal Point $cx: " << res.cx << std::endl;
        std::cout << "Principal Point $cy: " << res.cy << std::endl;
        std::cout << "==============================================\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
