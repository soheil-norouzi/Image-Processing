import numpy as np
import cv2

image = cv2.imread('/home/soheil/Pictures/Screenshots/Screenshot from 2025-04-13 23-25-39.png', cv2.IMREAD_GRAYSCALE)

#parameters
L = 5
h_L = L // 2
k_0 = 0.5 #threshold coefficients
k_1 = 1.5
k_2 = 0.4
k_3 = 2
C = 2 # contrast enhancement factor

glb_mean = np.mean(image) # global mean
glb_std = np.std(image) # global standard deviation

pad_image = np.pad(image, ((h_L, h_L), (h_L, h_L)) ,mode='reflect') # pad the image to handle borders
out_img= np.zeros_like(image, dtype=np.float32)

height , width = image.shape
for i in range (height): # process each pixel
    for j in range (width):
        window = pad_image[i:i+L , j:j+L] # extract the L x L window around the pixel
        hist , bins = np.histogram(window.flatten(), bins= 256, range=(0,256), density='True') # compute the histogram of the window to get Psxy
        r_i = np.arange(256)
        m_sxy = np.sum(hist * r_i) # local mean
        local_var_sqrt = np.sum(((r_i - m_sxy)**2) * hist ) # local variance
        if local_var_sqrt > 0:
            var_sxy = np.sqrt(local_var_sqrt)
        else:
            var_sxy = 0
        
        # enhancement conditions
        condition_mean = k_0 * glb_mean <= m_sxy <= k_1 * glb_mean
        condition_var= k_2 * glb_std <= var_sxy <= k_3 * glb_std

        # apply enhancement if conditions are met
        if condition_mean and condition_var:
            out_img[i, j] = C * image[i, j]
        else:
            out_img[i, j] = image[i, j]

outg_im = np.clip(out_img, 0 , 255)
out_img = out_img.astype(np.uint8)

cv2.imshow('Enhanced Image', out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()