# Bit-Plane Slicing in Python

## Overview
This project demonstrates **Bit-Plane Slicing**, a technique in image processing that extracts individual **bit-planes** from an 8-bit grayscale image. The higher bit-planes contain more structural information, while the lower bit-planes hold finer details.

The script:
- Loads a grayscale image.
- Extracts **8 bit-planes** using **bitwise operations**.
- Displays the **original image + 8 bit-planes + reconstructed**.

## Example Output
Below is an example of the **original image** and its **bit-plane decomposition** and **reconstructed image**:
### Original Image
![QWERTY](https://github.com/user-attachments/assets/48530897-c8b9-461c-8757-d0d50b6b2269)
<br>
### 8 Bit-Planes
![image](https://github.com/user-attachments/assets/2aab7af9-152a-4cb8-ac44-9efcf1b84ba6)
<br>  

### Reconstructed Image (Using Selected Bit-Planes)
![image](https://github.com/user-attachments/assets/c688f8f9-ff79-487b-acbd-b234d3ad8ac7)



# 🖼️ Local Histogram-Based Image Enhancement

This repository contains a simple Python script that performs **adaptive local histogram-based contrast enhancement** on grayscale images using NumPy and OpenCV.

---

## Overview

The algorithm enhances image contrast by scanning local neighborhoods around each pixel and applying contrast boosting only if the local statistics meet specific criteria relative to the global statistics.

This selective enhancement technique helps in revealing subtle details without over-amplifying noise, making it especially useful for images with uneven lighting or medical imaging.

---

##  How It Works

1. Load a grayscale image.
2. Pad the image to manage border effects.
3. For each pixel:
   - Extract an `L x L` neighborhood window.
   - Calculate the **local histogram**, **mean**, and **standard deviation**.
   - Compare the local statistics to scaled versions of the **global mean** and **global standard deviation**.
   - If conditions are satisfied, enhance the pixel intensity by multiplying it with contrast factor `C`.
4. Output the enhanced image.

---

## Parameters

| Parameter | Description |
|----------|-------------|
| `L` | Window size (must be odd; e.g., 5) |
| `k_0`, `k_1` | Range multipliers for global mean |
| `k_2`, `k_3` | Range multipliers for global standard deviation |
| `C` | Contrast enhancement multiplier |

You can fine-tune these parameters in the script to suit your specific use case.

---

## Dependencies

- Python 3.x
- OpenCV
- NumPy


pip install opencv-python numpy

# Frequency Domain Laplacian Image Enhancement

This repository contains a Python script that performs image enhancement using the Laplacian operator in the frequency domain. The script applies a Laplacian filter to the Fourier transform of an input image, enhances it, and visualizes the process with a 3x3 plot.

## Overview

The script implements image enhancement based on the following frequency domain Laplacian operation:
- Computes the 2D Fast Fourier Transform (FFT) of the input image.
- Applies a Laplacian filter \( H(u, v) = -4\pi^2 D^2(u, v) \), where \( D(u, v) \) is the distance from the center in the frequency domain.
- Enhances the image using \( g(x, y) = \mathcal{F}^{-1} [(1 + H(u, v)) F(u, v)] \).
- Visualizes the input image, magnitude spectrum, Laplacian, filtered spectrum, and enhanced image in a 3x3 grid.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

Install the required packages using pip:
pip install opencv-python numpy matplotlib

# Frequency Domain Image Enhancement with Mask (hp)

This code contains a Python script that performs image enhancement in the frequency domain using a Laplacian-based high-pass filter and a masking technique. The script enhances the image by subtracting a Laplacian-modified version and adding it back with a scaling factor.

## Overview

The script implements image enhancement based on the following equations:
- Laplacian filter: \( H(u, v) = -4\pi^2 \cdot 1e-4 \cdot D(u, v) \), where \( D(u, v) \) is the distance in the frequency domain.
- Masked image: \( g_{mask}(x, y) = f(x, y) - f_{LP}(x, y) \), where \( f_{LP}(x, y) = \mathcal{F}^{-1} [H(u, v) F(u, v)] \).
- Enhanced image: \( g(x, y) = f(x, y) + k g_{mask}(x, y) \), with \( k \) as a scaling factor.
- Visualizes the input image, masked image, and enhanced image in a 1x3 plot.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

Install the required packages using pip:
```bash
pip install opencv-python numpy matplotlib
