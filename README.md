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

Install them via pip:

```bash
pip install opencv-python numpy
