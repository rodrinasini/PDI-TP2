import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('img07.png', cv2.IMREAD_COLOR_RGB)
img_gris = cv2.imread('img01.png', cv2.IMREAD_GRAYSCALE)

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    """
    Muestra una imagen con matplotlib (función de la práctica).

    Recibe:
        img       : imagen (gris o RGB).
        new_fig   : si True crea una figura nueva.
        title     : título de la figura.
        color_img : True si la imagen es RGB, False si es en escala de grises.
        blocking  : si True la ventana bloquea la ejecución hasta cerrarla.
        colorbar  : si True agrega la barra de colores.
        ticks     : si True muestra los ejes.
    Devuelve:
        Nada.
    """
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

imshow(img_gris, title='Imagen original', color_img=False, blocking=True)

k = 2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))

fmg = cv2.morphologyEx(img_gris, cv2.MORPH_GRADIENT, kernel)
imshow(fmg, title='Gradiente Morfologico', color_img=False, blocking=True)

#img_blur = cv2.GaussianBlur(fmg, (3, 3), 0)
img_canny = cv2.Canny(fmg, threshold1=100, threshold2=150)
imshow(img_canny, title='Canny', color_img=False, blocking=True)

img_seg = cv2.morphologyEx(img_canny, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) 
imshow(img_seg, title='Segmentación Morfologica', color_img=False, blocking=True)

img_top_hat = cv2.morphologyEx(img_gris, kernel=kernel, op=cv2.MORPH_TOPHAT)
imshow(img_top_hat, title='Top Hat', color_img=False, blocking=True)


k = 11
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))

fo = cv2.morphologyEx(img_top_hat, cv2.MORPH_OPEN, kernel)
imshow(fo, title='Top Hat + Apertura', color_img=False, blocking=True)

T = (img_top_hat.min() + img_top_hat.max())/2                           # Inicializamos el valor de T
flag = False                                            # Inicializamos el flag para iterar
while not flag:
   g = img_top_hat >= T                                         # Umbralamos
   Tnext = 0.5*(np.mean(img_top_hat[g]) + np.mean(img_top_hat[~g]))     # Definimos nuevo umbral
   flag = np.abs(T-Tnext) < 0.5                         # Verificamos si hay cambios apreciables en el nuevo umbral
   T = Tnext
print(T)
_, img_th_aut = cv2.threshold(img_top_hat, thresh=T, maxval=255, type=cv2.THRESH_BINARY)
imshow(img_th_aut, title="Umbralado automático", colorbar=False)

T_otsu, img_th_otsu = cv2.threshold(img_top_hat, thresh=127, maxval=255, type=cv2.THRESH_OTSU)
print(T_otsu)

# --- Graficas -----------------------------------------
plt.figure()
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(img_top_hat, cmap="gray"), plt.title('Imagen Original')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_th_aut, cmap="gray"), plt.title(f'Umbral Automático: {T:5.2f}')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_th_otsu, cmap="gray"), plt.title(f'Umbral Otsu: {T_otsu:5.2f}')
plt.show(block=False)

#-------funcion aux--------
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((5,5), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection


def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh

img_fh = imfillhole(img_th_otsu)
plt.figure()
ax1 = plt.subplot(121); imshow(img_th_otsu, new_fig=False, title="Original", ticks=True)
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(img_fh, new_fig=False, title="Rellenado de Huecos")
plt.show(block=False)

#-------------Anotaciones--------

# Umbralado 
# Componentes conectadas - filtro por area
# Filtrar por la infomracion de la patente dada por la catedra
# Quedarnos con las componentes mas cercanas con su distancia