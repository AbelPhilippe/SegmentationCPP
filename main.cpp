#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    const string modelPath = "models/modnet_webcam.onnx";
    const int cameraIndex = argc > 1 ? stoi(argv[1]) : 0;

    constexpr int inputWidth = 512;
    constexpr int inputHeight = 512;

    // ONNX Runtime initialization.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MODNet");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);

    string inputName = inputNameAllocated.get();
    string outputName = outputNameAllocated.get();

    cout << "Model input: " << inputName << endl;
    cout << "Model output: " << outputName << endl;

    // Expected MODNet input dimensions: NCHW.
    vector<int64_t> inputShape = {
        1, 3, inputHeight, inputWidth
    };

    vector<float> inputTensorValues(
        3 * inputHeight * inputWidth
    );

    VideoCapture camera(cameraIndex, CAP_ANY);

    if (!camera.isOpened()) {
        cerr << "Could not open camera " << cameraIndex << endl;
        return 1;
    }

    camera.set(CAP_PROP_FRAME_WIDTH, 640);
    camera.set(CAP_PROP_FRAME_HEIGHT, 480);

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    const char* inputNames[] = { inputName.c_str() };
    const char* outputNames[] = { outputName.c_str() };

    Mat frame;

    while (camera.read(frame)) {
        if (frame.empty())
            continue;

        Mat resized, rgb, floatImage;
        resize(frame, resized, Size(inputWidth, inputHeight));
        cvtColor(resized, rgb, COLOR_BGR2RGB);
        rgb.convertTo(floatImage, CV_32FC3);

        vector<Mat> channels;
        split(floatImage, channels);

        for (int c = 0; c < 3; ++c) {
            float* destination =
                inputTensorValues.data() +
                c * inputWidth * inputHeight;

            for (int y = 0; y < inputHeight; ++y) {
                const float* source = channels[c].ptr<float>(y);

                for (int x = 0; x < inputWidth; ++x) {
                    destination[y * inputWidth + x] =
                        (source[x] - 127.5f) / 127.5f;
                }
            }
        }

        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size()
        );

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );

        if (!outputs[0].IsTensor()) {
            cerr << "Model output is not a tensor." << endl;
            break;
        }

        auto outputInfo = outputs[0].GetTensorTypeAndShapeInfo();
        auto outputShape = outputInfo.GetShape();

        if (outputShape.size() != 4 ||
            outputShape[0] != 1 ||
            outputShape[1] != 1) {
            cerr << "Unexpected output shape: ";
            for (auto d : outputShape)
                cerr << d << " ";
            cerr << endl;
            break;
        }

        const int outH = static_cast<int>(outputShape[2]);
        const int outW = static_cast<int>(outputShape[3]);

        const float* outputData =
            outputs[0].GetTensorData<float>();

        Mat alphaSmall(outH, outW, CV_32F,
                       const_cast<float*>(outputData));

        Mat alpha;
        resize(alphaSmall, alpha, frame.size());

        cv::max(alpha, 0.0, alpha);
        cv::min(alpha, 1.0, alpha);

        Mat mask;
        alpha.convertTo(mask, CV_8U, 255.0);

        Mat background(frame.size(), frame.type(),
                       Scalar(0, 180, 0));

        Mat alpha3;
        vector<Mat> alphaChannels(3, alpha);
        merge(alphaChannels, alpha3);

        Mat foregroundFloat, backgroundFloat;
        frame.convertTo(foregroundFloat, CV_32FC3);
        background.convertTo(backgroundFloat, CV_32FC3);

        Mat inverseAlpha = Scalar::all(1.0) - alpha3;
        Mat resultFloat =
            foregroundFloat.mul(alpha3) +
            backgroundFloat.mul(inverseAlpha);

        Mat result;
        resultFloat.convertTo(result, CV_8UC3);

        Mat maskBgr;
        cvtColor(mask, maskBgr, COLOR_GRAY2BGR);

        Mat display;
        hconcat(vector<Mat>{frame, result, maskBgr}, display);

        putText(display, "Original", Point(10, 25),
                FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 255), 2);

        putText(display, "Green background", Point(frame.cols + 10, 25),
                FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(255, 255, 255), 2);

        putText(display, "Alpha mask", Point(2 * frame.cols + 10, 25),
                FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 255), 2);

        imshow("MODNet - ONNX Runtime", display);

        int key = waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27)
            break;
    }

    camera.release();
    destroyAllWindows();

    return 0;
}
