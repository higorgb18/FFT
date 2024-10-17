import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#converte pra escala de cinza
img = Image.open('imagem.jpg').convert('RGB')
img_np = np.array(img)


#aplicação da FFT 2D pra cada canal de cor
def process_channel(channel, r):
    f_transform = np.fft.fft2(channel)
    f_shift = np.fft.fftshift(f_transform)

    #máscara pra eliminar altas frequências (filtro passa-baixa)
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1

    # aplicação da máscara no domínio da frequência
    f_shift_filtered = f_shift * mask

    #transformada inversa pro domínio espacial
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    #normalização imagem para valores entre 0 e 255
    img_back = np.uint8(255 * img_back / np.max(img_back))

    return img_back


#definição do raio de corte pra controlar a qualidade/compressao
r = 100

#processando cada canal de cor separadamente (R, G, B)
r_channel = process_channel(img_np[:, :, 0], r)
g_channel = process_channel(img_np[:, :, 1], r)
b_channel = process_channel(img_np[:, :, 2], r)

#reunir os canais processados
img_compressed = np.stack((r_channel, g_channel, b_channel), axis=-1)

#converte o formato de imagem e salva
Image.fromarray(img_compressed).save('imagemComprimidaFftCor.jpg')

plt.subplot(121), plt.imshow(img_np), plt.title('Imagem Original')
plt.subplot(122), plt.imshow(img_compressed), plt.title('Imagem Comprimida com FFT (Cor)')
plt.show()
