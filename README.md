# Procesamiento de Imágenes I - Trabajo Práctico N° 2

## Integrantes
* Gimbatti, Tomás
* Longo, Valentino
* Nasini, Rodrigo

## Resumen
Este proyecto aplica técnicas de procesamiento de imágenes para resolver dos problemas:

* **Problema 1 - Monedas y dados (`ejercicio_1.py`):** segmenta las monedas y los dados de la imagen `monedas.jpg` a partir de los bordes de Canny y operaciones morfológicas, los separa según su factor de forma, clasifica las monedas por su área mediante k-means (10 centavos, 1 peso y 50 centavos) y cuenta los puntos de la cara superior de cada dado.
* **Problema 2 - Patentes (`ejercicio_2.py`):** detecta la placa patente en las imágenes `img01.png` a `img12.png` resaltando los caracteres con un top-hat, buscando grupos de 6 objetos con forma de caracter alineados, y segmenta cada uno de sus caracteres.

## Pasos de Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/rodrinasini/PDI-TP2
   cd <nombre-del-directorio>
   ```

2. **Crear y activar un entorno virtual**:
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Instalar las librerías**:
   ```bash
   pip install requirements.txt
   ```

* `ejercicio_1.py` muestra el resultado de cada punto (A, B y C) y una figura final con las monedas clasificadas y el valor de cada dado.
* `ejercicio_2.py` muestra, para cada una de las 12 imágenes, las etapas de la detección de la patente y los caracteres segmentados; al cerrar cada ventana pasa a la siguiente imagen. Al final se muestra un resumen con las 12 patentes detectadas.

### Librerías utilizadas y sus versiones:
* Python 3.14.7
* numpy==2.5.4
* matplotlib==3.11.2
* opencv-python==5.0.0.93