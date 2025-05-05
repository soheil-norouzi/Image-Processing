import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('/media/soheil/Soheil/University/sem8/Image_processing_laplacian_Soheil_Norouzi/moon.png', cv2.IMREAD_GRAYSCALE)

fourier_transform = np.fft.fft2(image)
shift_fourier = np.fft.fftshift(fourier_transform)

rows , cols = image.shape
u = np.arange(cols) - cols//2
v = np.arange(rows) - rows//2
u , v = np.meshgrid(u,v)
D = u**2 + v**2
H = -4 * (np.pi**2) * 1e-5 * D

# compute the low-pass or laplacian
fourier_lp_freq = H * shift_fourier
fourier_lp_ishift = np.fft.ifftshift(fourier_lp_freq)
fourier_lp = np.fft.ifft2(fourier_lp_ishift)
fourier_lp = np.abs(fourier_lp)

g_mask = image.astype(np.float16) - fourier_lp.astype(np.float16) # compute the g_mask, g_mask = f(x, y) - f_LP(x, y)
g_mask = np.clip(g_mask, 0, 255).astype(np.uint8)

k = 0.45
g_enhance = image.astype(np.float16) - ( k * g_mask.astype(np.float16)) # enhance the image with: g(x, y) = f(x, y) + k * g_mask
g_enhance = np.clip(g_enhance, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Input Image")
axes[0].axis('off')

axes[1].imshow(g_mask, cmap='gray')
axes[1].set_title("Masked Image (g_mask)")
axes[1].axis('off')

axes[2].imshow(g_enhance, cmap='gray')
axes[2].set_title("Enhanced Image")
axes[2].axis('off')

plt.tight_layout()
plt.show()