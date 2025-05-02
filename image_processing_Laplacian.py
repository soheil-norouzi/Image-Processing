import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/media/soheil/Soheil/University/sem8/Image_processing_laplacian_Soheil_Norouzi/moon.png', cv2.IMREAD_GRAYSCALE)

fourier_transform = np.fft.fft2(image)
shift_fourier = np.fft.fftshift(fourier_transform)

rows, cols = image.shape
u = np.arange(cols) - rows//2
v = np.arange(rows) -cols//2
u , v = np.meshgrid(u,v)
D = u**2 + v**2
H =  -4 * (np.pi**2) * 1e-5 * D # Laplacian filter in frequency domain

filtered_freq = H * shift_fourier # filtering in frequency domain

# compute the inverse fft to get enhanced image
filtered_invshift = np.fft.ifftshift(filtered_freq)
image_back = np.fft.ifft2(filtered_invshift)
image_back = np.abs(image_back)
enhanced_image = np.clip(image_back, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Input Image")
axes[0].axis('off')

axes[1].imshow(enhanced_image, cmap='gray')
axes[1].set_title("Enhanced Image")
axes[1].axis('off')

plt.tight_layout()
plt.show()