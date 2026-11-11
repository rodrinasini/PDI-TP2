import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------- FUNCIONES AUXILIARES ----------------------------------------

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


def rellenar_agujeros(img):
    """
    Rellena los huecos de los objetos de una imagen binaria (floodFill desde el fondo).

    Recibe:
        img : imagen binaria uint8 (0 = fondo, 255 = objeto).
    Devuelve:
        img_fh : imagen binaria con los huecos interiores rellenos.
    """
    img_flood_fill = img.copy().astype("uint8")
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_flood_fill, mask, (0, 0), 255)        # Pinto el fondo desde la esquina
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)    # Lo que quedó sin pintar son los huecos
    img_fh = img | img_flood_fill_inv                       # Sumo los huecos al objeto
    return img_fh


def factor_forma_inv(cnt):
    """
    Calcula la inversa del factor de forma: 1/Fp = P² / A.

    Recibe:
        cnt : contorno de un objeto (salida de cv2.findContours).
    Devuelve:
        float : P²/A  (círculo ≈ 12.57 teórico / ~14 digital, cuadrado ≈ 16).
    """
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    return perimetro**2 / area


# ------------------------------------- PUNTO A ----------------------------------

def segmentacion(img):
    """
    Segmenta los objetos de la imagen y los separa en monedas y dados según su factor de forma.

    Recibe:
        img : imagen original en BGR (como la lee cv2.imread).
    Devuelve:
        img_ab  : imagen RGB con monedas en verde y dados en rojo.
        monedas : lista de diccionarios {"cnt", "area", "fp_inv", "bbox"} de cada moneda.
        dados   : lista de diccionarios con la misma estructura, de cada dado.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
    img_canny = cv2.Canny(img_blur, threshold1=30, threshold2=90)

    K = 31
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
    img_dil = cv2.dilate(img_canny, kernel, iterations=1)      
    img_fh = rellenar_agujeros(img_dil)                       
    img_ero = cv2.erode(img_fh, kernel)            
    img_seg = cv2.morphologyEx(img_ero, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) 

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_seg, connectivity=8, ltype=cv2.CV_32S)
    labels_norm = np.uint8((255/(num_labels-1)) * labels)
    im_color = cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET)
    im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
    im_color[labels == 0] = 0

    TH_FP_INV = 15
    contours, hierarchy = cv2.findContours(img_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    monedas = []
    dados = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        fp_inv = factor_forma_inv(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        objeto = {"cnt": cnt, "area": area, "fp_inv": fp_inv, "bbox": (x, y, w, h)}
        if fp_inv < TH_FP_INV:
            monedas.append(objeto)
        else:
            dados.append(objeto)

    img_ab = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for m in monedas:
        cv2.drawContours(img_ab, [m["cnt"]], -1, (0, 255, 0), 10)
    for d in dados:
        cv2.drawContours(img_ab, [d["cnt"]], -1, (255, 0, 0), 10)
    imshow(img_ab, color_img=True, title=f"PUNTO A - Monedas (verde): {len(monedas)} | Dados (rojo): {len(dados)}")

    return img_ab, monedas, dados


# --------------------------------------- PUNTO B -----------------------------------

def clasificacion_monedas(img_rgb, monedas):
    """
    Clasifica cada moneda según su área (10 centavos, 1 peso o 50 centavos) y las cuenta.

    Recibe:
        img_rgb : imagen RGB sobre la que se dibuja el resultado.
        monedas : lista de diccionarios de monedas (salida de segmentacion).
                A cada diccionario se le agrega la clave "tipo".
    Devuelve:
        img_b           : imagen RGB con cada moneda pintada y etiquetada.
        conteo_monedas  : diccionario {tipo: cantidad}.
        colores_monedas : diccionario {tipo: color RGB} usado para dibujar.
        total_dinero    : suma de dinero en pesos.
    """
    TH_AREA_10C_1P = 75000
    TH_AREA_1P_50C = 95000

    conteo_monedas = {"10 centavos": 0, "1 peso": 0, "50 centavos": 0}
    colores_monedas = {"10 centavos": (0, 255, 255), "1 peso": (255, 0, 255), "50 centavos": (255, 255, 0)}

    for m in monedas:
        if m["area"] < TH_AREA_10C_1P:
            m["tipo"] = "10 centavos"
        elif m["area"] < TH_AREA_1P_50C:
            m["tipo"] = "1 peso"
        else:
            m["tipo"] = "50 centavos"
        conteo_monedas[m["tipo"]] += 1

    total_dinero = conteo_monedas["10 centavos"]*0.10 + conteo_monedas["1 peso"]*1 + conteo_monedas["50 centavos"]*0.50

    img_b = img_rgb.copy()
    for m in monedas:
        cv2.drawContours(img_b, [m["cnt"]], -1, colores_monedas[m["tipo"]], 10)
        x, y, w, h = m["bbox"]
        cv2.putText(img_b, m["tipo"], (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.8, colores_monedas[m["tipo"]], 5)
    titulo = " | ".join([f"{t}: {c}" for t, c in conteo_monedas.items()])
    imshow(img_b, color_img=True, title=f"PUNTO B - {titulo}")
    
    return img_b, conteo_monedas, colores_monedas, total_dinero


# ----------------------------------------- PUNTO C ------------------------------------

def clasificacion_dados(img_gray, img_rgb, dados):
    """
    Cuenta los puntos (valor) de la cara superior de cada dado.

    Recibe:
        img_gray : imagen original en escala de grises (de donde se buscan los puntos).
        img_rgb  : imagen RGB sobre la que se dibuja el resultado.
        dados    : lista de diccionarios de dados (salida de segmentacion).
                    A cada diccionario se le agrega la clave "valor".
    Devuelve:
        img_c      : imagen RGB con cada dado etiquetado con su valor.
        suma_dados : suma de los valores de todos los dados.
    """
    TH_PUNTO = 80      
    TH_FP_INV_PUNTO = 17    

    img_c = img_rgb.copy()
    for d in dados:
        x, y, w, h = d["bbox"]

        mask = np.zeros_like(img_gray)
        cv2.drawContours(mask, [d["cnt"]], -1, 255, -1)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        mask = mask[y:y+h, x:x+w]
        dado = cv2.GaussianBlur(img_gray[y:y+h, x:x+w], ksize=(5, 5), sigmaX=0)

        _, dado_th = cv2.threshold(dado, thresh=TH_PUNTO, maxval=255, type=cv2.THRESH_BINARY_INV)
        dado_th = cv2.bitwise_and(dado_th, mask)
        dado_fh = rellenar_agujeros(dado_th)
        dado_op = cv2.morphologyEx(dado_fh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        cnts_puntos, _ = cv2.findContours(dado_op, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_puntos = [c for c in cnts_puntos if cv2.contourArea(c) > 0]
        area_max = max([cv2.contourArea(c) for c in cnts_puntos])
        puntos = []
        for c in cnts_puntos:
            a = cv2.contourArea(c)
            f = factor_forma_inv(c)
            if a > 0.5*area_max and f < TH_FP_INV_PUNTO:
                puntos.append(c)
        d["valor"] = len(puntos)

        cv2.drawContours(img_c, [d["cnt"]], -1, (255, 0, 0), 10)
        cv2.putText(img_c, f"Dado: {d['valor']}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 5)

    suma_dados = sum([d["valor"] for d in dados])

    titulo = " | ".join([f"Dado {ii+1}: {d['valor']}" for ii, d in enumerate(dados)])
    imshow(img_c, color_img=True, title=f"PUNTO C - {titulo} | Suma: {suma_dados}")

    return img_c, suma_dados


def ejercicio_1(ruta_imagen):
    """
    Ejecuta los tres puntos del Problema 1 en orden (A -> B -> C) y muestra el resultado final.

    Recibe:
        ruta_imagen : ruta (relativa) de la imagen de monedas y dados, por ej. 'monedas.jpg'.
    Devuelve:
        monedas : lista de monedas, cada una con su "tipo".
        dados   : lista de dados, cada uno con su "valor".
    """
    img = cv2.imread(ruta_imagen)                       # BGR
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_ab, monedas, dados = segmentacion(img)

    img_b, conteo_monedas, colores_monedas, total_dinero = clasificacion_monedas(img_rgb, monedas)

    img_c, suma_dados = clasificacion_dados(img_gray, img_b, dados)

    titulo = " | ".join([f"{t}: {c}" for t, c in conteo_monedas.items()]) + f" | Total: ${total_dinero:.2f}"
    titulo += "\n" + " | ".join([f"Dado {ii+1}: {d['valor']}" for ii, d in enumerate(dados)])
    imshow(img_c, color_img=True, title=f"RESULTADO FINAL\n{titulo}", blocking=True)

    return monedas, dados

# -------------------------------------------- EJECUCIÓN ---------------------------------------------

monedas, dados = ejercicio_1('monedas.jpg')
