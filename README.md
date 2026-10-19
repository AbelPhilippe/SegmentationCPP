# Real-Time Person Segmentation with C++ and OpenCV

Real-time webcam foreground segmentation and background replacement using **MODNet**, **ONNX Runtime**, and **OpenCV**.

The application captures video from a webcam, estimates an alpha mask for the foreground person, and replaces the background with a solid green color.

## Features

- Real-time webcam capture using OpenCV.
- Person segmentation using a pretrained MODNet ONNX model.
- ONNX inference through ONNX Runtime's C++ API.
- Alpha-mask generation and foreground/background blending.
- Side-by-side visualization of the original frame, segmented result, and mask.
- CPU inference support.

## Technologies

| Technology | Purpose |
|---|---|
| C++17 | Application implementation |
| OpenCV 4.6.0 | Webcam capture, image processing, and visualization |
| ONNX Runtime 1.22.0 | Neural network inference |
| MODNet | Portrait foreground matting |
| ONNX | Model interchange format |
| Operating systems | Linux, Windows, and macOS (with platform-specific setup) |

## Project Structure

```text
TEST/
├── main.cpp
├── segmentation
├── models/
│   └── modnet_webcam.onnx
└── onnxruntime/
    ├── include/
    └── lib/
```

The `segmentation` executable is generated during compilation.

## Requirements

- **Operating system:** Linux, Windows 10/11, or macOS.
- **Architecture:** Supported 64-bit CPU architecture for the selected ONNX Runtime package.
- **Compiler:** C++17-compatible compiler, such as GCC, Clang, or MSVC.
- **OpenCV:** Development libraries compatible with the target platform.
- **ONNX Runtime:** C++ libraries compatible with the operating system and architecture.
- **Webcam:** A working camera supported by OpenCV.
- **Model:** A compatible MODNet ONNX model.

Platform-specific installation and compilation instructions may be required.


## Installation

### Windows

Install a C++ compiler, OpenCV, and ONNX Runtime for Windows.
Configure the include directories, library directories, and runtime DLL paths
for your development environment.

### macOS

Install a C++ compiler, OpenCV, and ONNX Runtime using a supported package
manager or the official binary distributions. Configure the compiler and
runtime library paths accordingly.

### 1. Install build dependencies (Linux)

```bash
sudo apt update
sudo apt install build-essential pkg-config libopencv-dev wget
```

Verify OpenCV:

```bash
pkg-config --modversion opencv4
```

### 2. Download ONNX Runtime

From the project directory:

```bash
cd ~/Projects/C-general/TEST

wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz

tar -xzf onnxruntime-linux-x64-1.22.0.tgz

mv onnxruntime-linux-x64-1.22.0 onnxruntime
```

Verify the required files:

```bash
ls onnxruntime/include/onnxruntime_cxx_api.h
ls onnxruntime/lib/libonnxruntime.so
```

### 3. Obtain the MODNet model

Create the model directory:

```bash
mkdir -p models
```

Place a compatible ONNX model at:

```text
models/modnet_webcam.onnx
```

The model must be compatible with the preprocessing and input/output tensor layouts expected by `main.cpp`.

MODNet project: https://github.com/yakhyo/modnet

ONNX Runtime releases: https://github.com/microsoft/onnxruntime/releases

**Note:** Verify that the model download succeeds and that the file is a valid ONNX model before running the application.

## Build

Compile the application:

```bash
g++ -std=c++17 -O3 main.cpp -o segmentation \
    -I./onnxruntime/include \
    -L./onnxruntime/lib \
    -Wl,-rpath,'$ORIGIN/onnxruntime/lib' \
    $(pkg-config --cflags opencv4) \
    $(pkg-config --libs opencv4) \
    -lonnxruntime
```

This command links OpenCV and ONNX Runtime and embeds a runtime library search path relative to the executable.

## Run

Start the application with the default webcam:

```bash
./segmentation 0
```

To select another camera:

```bash
./segmentation 1
```

Press `q` or `Esc` to close the application.

## How It Works

1. **Capture:** OpenCV captures a webcam frame.
2. **Preprocessing:** The frame is resized, converted from BGR to RGB, normalized to the expected range, and arranged into a tensor.
3. **Inference:** ONNX Runtime executes the MODNet model.
4. **Mask generation:** The model's predicted alpha mask is resized to match the original frame.
5. **Background replacement:** The foreground and a solid green background are blended using the alpha mask.
6. **Visualization:** The application displays the original frame, the composite image, and the grayscale alpha mask.

## Troubleshooting

### ONNX model not found

Check that the model exists:

```bash
ls -lh models/modnet_webcam.onnx
```

Run the executable from the project directory, or update the model path in `main.cpp`.

### ONNX Runtime library not found

Check the library:

```bash
ls onnxruntime/lib/libonnxruntime.so
```

Inspect runtime dependencies:

```bash
ldd ./segmentation
```

### Webcam cannot be opened

List video devices:

```bash
ls -l /dev/video*
```

Check available camera devices:

```bash
v4l2-ctl --list-devices
```

Install the utility if needed:

```bash
sudo apt install v4l-utils
```

### Inference fails

Check the model's input and output metadata. The example implementation assumes an input tensor shaped `[1, 3, 512, 512]` and an alpha output shaped `[1, 1, H, W]`.

If the model uses a different tensor layout, input resolution, normalization scheme, or output format, the preprocessing and postprocessing must be adjusted.

### Low frame rate

- Reduce the webcam resolution.
- Use an appropriate model input resolution supported by the selected model.
- Avoid unnecessary image copies.
- Measure inference time separately from rendering.
- Consider a model optimized for real-time portrait matting.

## Limitations

- MODNet is designed primarily for portrait matting, not arbitrary-object segmentation.
- Hair, motion blur, poor lighting, and complex backgrounds may affect mask quality.
- Performance depends on CPU capabilities, model resolution, and implementation.
- The application requires a compatible ONNX model; a successful C++ compilation does not guarantee successful inference.

## Future Improvements

- Replace the solid background with a custom image or video.
- Add background blur.
- Display FPS and inference latency (Low FPS at the moment)
- Optimize preprocessing and memory allocation.
- Explore GPU inference where supported.
- Evaluate alternative segmentation models for different performance requirements.

## About License

Check the licenses of the selected MODNet weights, ONNX Runtime, and OpenCV before redistributing the application or model.