import re


def detectar_placa_desde_imagen(image_path):
    try:
        image = cv2.imread(image_path)

        if image is None:
            print('No se pudo leer la imagen')
            return None

        # Detectar placas con YOLO
        resultados = modelo_placas(image_path)

        for resultado in resultados:
            cajas = resultado.boxes

            for caja in cajas:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])

                # Recortar la placa
                placa = image[y1:y2, x1:x2]

                if placa.size == 0:
                    continue

                print(f'Placa detectada en coordenadas: {x1}, {y1}, {x2}, {y2}')

                # Convertir a gris
                gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

                # Agrandar imagen
                gray = cv2.resize(
                    gray,
                    None,
                    fx=4,
                    fy=4,
                    interpolation=cv2.INTER_CUBIC
                )

                # Mejorar imagen
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                gray = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]

                # OCR
                texto = pytesseract.image_to_string(
                    gray,
                    config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                )

                texto = limpiar_texto(texto)

                print(f'Texto OCR detectado: {texto}')

                # Validar formato
                if re.match(r'^[A-Z0-9]{6,8}$', texto):
                    return texto

        return None

    except Exception as e:
        print(f'Error OCR con YOLO: {e}')
        return None
