import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#converte pra escala de cinza
img = Image.open('imagem.jpg').convert('L')
img_np = np.array(img)

#aplicação da FFT 2D
f_transform = np.fft.fft2(img_np)
f_shift = np.fft.fftshift(f_transform)

#definição do espectro de magnitude e fase
magnitude_spectrum = np.abs(f_shift)
phase_spectrum = np.angle(f_shift)

#máscara pra descartar frequências altas
rows, cols = img_np.shape
crow, ccol = rows // 2, cols // 2

#definição do raio de corte pra controlar a qualidade/compressao
r = 100
mask = np.zeros((rows, cols), np.uint8)
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
mask[mask_area] = 1

#aplicação da máscara pra remover as altas frequências
f_shift_filtered = f_shift * mask

#transformada inversa pro domínio espacial
f_ishift = np.fft.ifftshift(f_shift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

#normalização imagem para valores entre 0 e 255
img_back = np.uint8(255 * img_back / np.max(img_back))

Image.fromarray(img_back).save('imagemComprimidaFft.jpg')

plt.subplot(121), plt.imshow(img_np, cmap='gray'), plt.title('Imagem Original')
plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('Imagem Comprimida com FFT')
plt.show()
