
import re
import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)


def limpiar_texto(texto):
    texto = texto.upper()
    texto = texto.replace(' ', '')
    texto = re.sub(r'[^A-Z0-9]', '', texto)
    return texto


def detectar_placa_desde_imagen(image_path):
    try:
        # Leer imagen
        image = cv2.imread(image_path)

        if image is None:
            return None

        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mejorar contraste
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # OCR
        resultados = reader.readtext(gray)

        textos = []
        for r in resultados:
            texto = r[1]
            texto = limpiar_texto(texto)
            textos.append(texto)

        # Buscar posibles placas
        for texto in textos:
            if re.match(r'^[A-Z0-9]{6,8}$', texto):
                return texto

        return None

    except Exception as e:
        print(f'Error OCR: {e}')
        return None
