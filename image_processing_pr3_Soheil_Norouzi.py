# Project 3
# Soheil Norouzi

import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image = cv2.imread("/media/soheil/Soheil/University/sem8/image_processing_pr3_Soheil_Norouzi/Screenshot from 2025-04-25 22-58-48.png", cv2.IMREAD_GRAYSCALE)

#a.
laplacian_kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]], dtype = np.float32)
laplacian = cv2.filter2D(input_image, cv2.CV_64F, laplacian_kernel)
laplacian_image = input_image.astype(np.float64) - laplacian
laplacian_image = np.clip(laplacian_image, 0, 256).astype(np.uint8)

#b
sum_image_a_b = cv2.add(input_image, laplacian_image)

#c
soble_x = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize = 3)
soble_y = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize = 3)
sobel_gradient = cv2.magnitude(soble_x, soble_y)

#d
box_filter = np.ones((5,5), dtype = np.float32) / 25
sobol_smooth = cv2.filter2D(sobel_gradient, -1, box_filter)

#e
laplacian_norm = laplacian_image.astype(np.float64) / 255
sobel_norm = sobol_smooth.astype(np.float64) / 255
multiplied_image = laplacian_norm * sobel_norm
multiplied_image = (multiplied_image * 255).astype(np.uint8)

#f 
sum_image_a_f = cv2.add(input_image, multiplied_image)

#g
c = 1
gamma = 0.5
normalized_img = sum_image_a_f.astype(np.float64) / 255
gamma_pic = c * np.power(normalized_img, gamma)
gamma_pic = (gamma_pic * 255).astype(np.uint8)

# plot the results
fig, axes = plt.subplots(3,3, figsize = (10, 10))
axes = axes.ravel()

images = [
    input_image, laplacian_image, sum_image_a_b,
    sobel_gradient, sobol_smooth, multiplied_image,
    sum_image_a_f, gamma_pic
]
titles = [
    "Input Image", "Step b: Laplacian Image", "Step c: Sum (a+b)",
    "Step d: Sobel Gradient", "Step e: Sobel Smoothed", "Step f: Multiply (e*b)",
    "Step g: Sum (a+f)", "Step h: Gamma Corrected"
]

for i in range(len(images)):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(titles[i])
    axes[i].axis('off')
axes[8].axis('off')

plt.tight_layout()
plt.show()

