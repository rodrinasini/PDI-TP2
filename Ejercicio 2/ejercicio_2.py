import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- Parámetros --------------------------------------------------------------

K_TOPHAT = 15                   
UMBRALES = range(30, 200, 5)
ALTO_MIN, ALTO_MAX = 8, 30   
AREA_MIN = 15                   
RA_MIN, RA_MAX = 1.5, 3.0      



# --- Funciones ---------------------------------------------------------------

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    """
    Muestra una imagen con matplotlib (función de la práctica).

    Recibe:  la imagen y opciones de visualización (título, color, bloqueo, etc.).
    Devuelve: nada.
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


def candidatos_caracteres(img_bin):
    """
    Busca los objetos de una imagen binaria que tienen forma de caracter.

    Recibe:
        img_bin : imagen binaria (0 / 255).
    Devuelve:
        candidatos : lista de tuplas (x, y, ancho, alto) de cada objeto con forma de caracter.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8, ltype=cv2.CV_32S)
    candidatos = []
    for i in range(1, num_labels):         
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if ALTO_MIN <= h <= ALTO_MAX and a >= AREA_MIN and RA_MIN <= h/w <= RA_MAX:
            candidatos.append((int(x), int(y), int(w), int(h)))
    return candidatos


def caracteres_patente(candidatos):
    """
    Se queda con los candidatos que tienen al menos 5 vecinos parecidos (los otros 5 caracteres).
    Un vecino es otro candidato que:
        * tiene un alto parecido     (diferencia menor al 30% del alto)
        * está a la misma altura     (diferencia en y menor a un alto)
        * está cerca horizontalmente (a menos de 5 altos)

    Recibe:
        candidatos : lista de tuplas (x, y, ancho, alto).
    Devuelve:
        caracteres : lista de tuplas (x, y, ancho, alto), ordenada de izquierda a derecha.
    """
    caracteres = []
    for (x, y, w, h) in candidatos:
        vecinos = 0
        for (x2, y2, w2, h2) in candidatos:
            if abs(h2 - h) < 0.3*h and abs(y2 - y) < h and abs(x2 - x) < 5*h:
                vecinos += 1
        if vecinos >= 6:                      
            caracteres.append((x, y, w, h))
    return sorted(caracteres)                  


def detectar_patente(img_gray):
    """
    PUNTO A: detecta la patente.

    Recibe:
        img_gray : imagen en escala de grises.
    Devuelve:
        img_tophat : resultado del Top-Hat.
        img_bin    : Top-Hat umbralado con el umbral elegido.
        umbral     : umbral con el que se encontró la patente (None si no se encontró).
        rect       : rectángulo de la patente (x, y, ancho, alto) (None si no se encontró).
    """
    
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (K_TOPHAT, K_TOPHAT))
    img_tophat = cv2.morphologyEx(img_gray, kernel=se, op=cv2.MORPH_TOPHAT)

    
    for umbral in UMBRALES:
        _, img_bin = cv2.threshold(img_tophat, umbral, 255, cv2.THRESH_BINARY)
        caracteres = caracteres_patente(candidatos_caracteres(img_bin))
        if len(caracteres) == 6:
            
            x_ini = caracteres[0][0]
            x_fin = caracteres[5][0] + caracteres[5][2]
            y_ini = min([c[1] for c in caracteres])
            y_fin = max([c[1] + c[3] for c in caracteres])
            margen = caracteres[0][3] // 2          
            rect = (x_ini - margen, y_ini - margen, (x_fin - x_ini) + 2*margen, (y_fin - y_ini) + 2*margen)
            return img_tophat, img_bin, umbral, rect

    return img_tophat, img_bin, None, None


def segmentar_caracteres(img_bin, rect):
    """
    PUNTO B: segmenta los caracteres dentro de la patente.

    Recibe:
        img_bin : Top-Hat umbralado (salida de detectar_patente).
        rect    : rectángulo de la patente (x, y, ancho, alto).
    Devuelve:
        patente_bin : recorte binario de la patente.
        letras      : lista con el recorte binario de cada caracter, de izquierda a derecha.
        cajas       : lista de tuplas (x, y, ancho, alto) de cada caracter, relativas a la patente.
    """
    x, y, w, h = rect
    patente_bin = img_bin[y:y+h, x:x+w]
    cajas = caracteres_patente(candidatos_caracteres(patente_bin))
    letras = []
    for (xc, yc, wc, hc) in cajas:
        letras.append(patente_bin[yc:yc+hc, xc:xc+wc])
    return patente_bin, letras, cajas

# --- Programa principal ------------------------------------------------------

resultados = []
for n_img in range(1, 13):
    nombre = f"img{n_img:02d}.png"
    img = cv2.imread(nombre)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_tophat, img_bin, umbral, rect = detectar_patente(img_gray)
    if rect is None:
        print(f"{nombre}: no se encontró la patente")
        continue

    x, y, w, h = rect
    patente = img_rgb[y:y+h, x:x+w]
    img_rect = img_rgb.copy()
    cv2.rectangle(img_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)

    patente_bin, letras, cajas = segmentar_caracteres(img_bin, rect)
    patente_cajas = patente.copy()
    for (xc, yc, wc, hc) in cajas:
        cv2.rectangle(patente_cajas, (xc, yc), (xc+wc, yc+hc), (255, 0, 0), 1)

    print(f"{nombre}: umbral = {umbral} | patente en {rect} | caracteres segmentados = {len(letras)}")
    resultados.append((nombre, patente))

    plt.figure(figsize=(14, 7))
    plt.suptitle(f"{nombre} - umbral {umbral}")
    plt.subplot(2, 3, 1), imshow(img_gray, new_fig=False, title="Original")
    plt.subplot(2, 3, 2), imshow(img_tophat, new_fig=False, title="1. Top-Hat")
    plt.subplot(2, 3, 3), imshow(img_bin, new_fig=False, title=f"2. Umbral {umbral}")
    plt.subplot(2, 3, 4), imshow(img_rect, new_fig=False, color_img=True, title="A. Patente detectada")
    plt.subplot(2, 3, 5), imshow(patente_cajas, new_fig=False, color_img=True, title="A. Patente segmentada")
    plt.subplot(2, 3, 6), imshow(patente_bin, new_fig=False, title=f"B. Caracteres: {len(letras)}")
    plt.show(block=False)

    plt.figure(figsize=(8, 2))
    plt.suptitle(f"{nombre} - Punto B: caracteres segmentados")
    for i in range(len(letras)):
        plt.subplot(1, len(letras), i+1), imshow(letras[i], new_fig=False, title=f"{i+1}")
    plt.show(block=True)

print(f"\nPatentes detectadas: {len(resultados)} de 12")
plt.figure(figsize=(12, 6))
for i in range(len(resultados)):
    plt.subplot(3, 4, i+1), imshow(resultados[i][1], new_fig=False, color_img=True, title=resultados[i][0])
plt.suptitle(f"Patentes detectadas: {len(resultados)} de 12")
plt.show(block=True)
