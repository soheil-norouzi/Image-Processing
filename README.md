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

