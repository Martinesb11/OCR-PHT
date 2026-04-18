import re
import cv2
import pytesseract


def limpiar_texto(texto):
    texto = texto.upper()
    texto = texto.replace(' ', '')
    texto = texto.replace('-', '')
    texto = texto.replace('\n', '')
    texto = re.sub(r'[^A-Z0-9]', '', texto)
    return texto


def detectar_placa_desde_imagen(image_path):
    try:
        image = cv2.imread(image_path)

        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reducir ruido
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Detectar bordes
        edges = cv2.Canny(gray, 30, 200)

        # Buscar contornos
        contornos, _ = cv2.findContours(
            edges.copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:20]

        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            aproximacion = cv2.approxPolyDP(contorno, 0.018 * perimetro, True)

            # Buscar rectángulos de 4 lados
            if len(aproximacion) == 4:
                x, y, w, h = cv2.boundingRect(contorno)

                # Filtrar tamaños pequeños
                if w < 80 or h < 25:
                    continue

                placa = gray[y:y + h, x:x + w]

                # Agrandar recorte
                placa = cv2.resize(
                    placa,
                    None,
                    fx=4,
                    fy=4,
                    interpolation=cv2.INTER_CUBIC
                )

                # Mejorar imagen
                placa = cv2.GaussianBlur(placa, (5, 5), 0)
                placa = cv2.threshold(
                    placa,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]

                texto = pytesseract.image_to_string(
                    placa,
                    config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                )

                texto = limpiar_texto(texto)

                print(f'Texto OCR detectado: {texto}')

                if re.match(r'^[A-Z0-9]{6,8}$', texto):
                    return texto

        return None

    except Exception as e:
        print(f'Error OCR: {e}')
        return None
