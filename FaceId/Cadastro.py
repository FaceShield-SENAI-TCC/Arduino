import cv2
import os
import time
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ====================== CONFIGURAÇÕES AVANÇADAS ======================
DATABASE_DIR = "facial_database"
CAPTURE_DURATION = 12
TARGET_FACES = 60
MIN_FACE_SIZE = (120, 120)
FACE_DETECTOR = "dnn"  # Alterado para padrão mais robusto

# Parâmetros de qualidade
MIN_SHARPNESS = 100
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 200
MIN_FACE_CONFIDENCE = 0.9


# ====================== INICIALIZAÇÃO DE MODELOS ======================
def initialize_detector(detector_type):
    """Inicializa o detector facial escolhido"""
    if detector_type == "dnn":
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "deploy.prototxt"

        if not os.path.exists(model_file) or not os.path.exists(config_file):
            download_dnn_model()

        net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        return {"type": "dnn", "net": net}

    elif detector_type == "mtcnn":
        try:
            from mtcnn import MTCNN
            return {"type": "mtcnn", "detector": MTCNN()}
        except ImportError:
            print("MTCNN não disponível. Usando DNN como fallback.")
            return initialize_detector("dnn")

    else:  # Haar Cascade padrão
        cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return {"type": "haar", "detector": cv2.CascadeClassifier(cascade_file)}


# ====================== FUNÇÕES AUXILIARES AVANÇADAS ======================
def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '', name.replace(' ', '_'))


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])


def enhance_face_image(face_img):
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return enhanced


def face_quality_score(face_img):
    if face_img.size == 0:
        return 0

    sharpness = calculate_sharpness(face_img)
    sharp_score = min(1, sharpness / 200) * 50

    brightness = calculate_brightness(face_img)
    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
        bright_score = 0
    else:
        bright_score = (1 - abs(brightness - 120) / 80) * 30

    return sharp_score + bright_score


# ====================== DETECÇÃO FACIAL AVANÇADA ======================
def detect_faces(frame, detector):
    faces = []  # Inicializa lista vazia

    if detector["type"] == "dnn":
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        detector["net"].setInput(blob)
        detections = detector["net"].forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > MIN_FACE_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype("int"))

    elif detector["type"] == "mtcnn":
        results = detector["detector"].detect_faces(frame)
        for res in results:
            if res['confidence'] > MIN_FACE_CONFIDENCE:
                x, y, w, h = res['box']
                faces.append([x, y, x + w, y + h])

    else:  # Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = detector["detector"].detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=MIN_FACE_SIZE
        )
        faces = [[x, y, x + w, y + h] for (x, y, w, h) in haar_faces]

    return faces


# ====================== CAPTURA DE FOTOS ======================
def capture_faces(user_dir, user_name, safe_name):
    detector = initialize_detector(FACE_DETECTOR)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Erro ao abrir a câmera!")

    print(f"\nPosicione seu rosto. Captura iniciando em 3 segundos...")
    time.sleep(3)

    start_time = time.time()
    captured_count = 0
    frame_count = 0
    faces = []  # Inicializa faces vazio

    executor = ThreadPoolExecutor(max_workers=4)
    futures = []

    print("Capturando... (Pressione ESC para encerrar antecipadamente)")

    while (time.time() - start_time) < CAPTURE_DURATION and captured_count < TARGET_FACES:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Detectar rostos a cada 3 frames
        if frame_count % 3 == 0:
            faces = detect_faces(frame, detector)

        # Processar cada rosto detectado
        for (x1, y1, x2, y2) in faces:  # Agora faces sempre definido
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            margin = int((x2 - x1) * 0.15)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size > 0:
                quality = face_quality_score(face_img)
                quality_text = f"Qual: {int(quality)}%"
                cv2.putText(display_frame, quality_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if quality > 60 and captured_count < TARGET_FACES:
                    future = executor.submit(save_face_image, face_img, user_dir, safe_name, captured_count, quality)
                    futures.append(future)
                    captured_count += 1
                    cv2.putText(display_frame, "CAPTURADO!", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elapsed = time.time() - start_time
        remaining = max(0, CAPTURE_DURATION - int(elapsed))
        status = f"{user_name} | Tempo: {remaining}s | Capturadas: {captured_count}/{TARGET_FACES}"
        cv2.putText(display_frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Cadastro Facial', display_frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)

    print(f"\nCaptura concluída! {captured_count} faces de alta qualidade salvas.")
    return captured_count


def save_face_image(face_img, user_dir, safe_name, index, quality):
    try:
        enhanced = enhance_face_image(face_img)
        timestamp = datetime.now().strftime("%H%M%S%f")
        filename = f"{safe_name}_{index:03d}_{timestamp}_{int(quality)}.jpg"
        filepath = os.path.join(user_dir, filename)
        cv2.imwrite(filepath, enhanced)
        print(f"Salvo: {filename} (Qualidade: {int(quality)}%)")
    except Exception as e:
        print(f"Erro ao salvar face: {str(e)}")


def download_dnn_model():
    import urllib.request
    print("Baixando modelo DNN...")

    files = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }

    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Baixando {filename}...")
            urllib.request.urlretrieve(url, filename)

    print("Download de modelos DNN completo!")


def main():
    print("======== SISTEMA DE CADASTRO FACIAL AVANÇADO ========")
    print(f"Configuração: {TARGET_FACES} faces em {CAPTURE_DURATION} segundos")
    print(f"Detector: {FACE_DETECTOR.upper()}\n")

    user_name = input("Digite o nome completo do usuário: ").strip()
    if not user_name:
        print("Nome inválido!")
        return

    safe_name = sanitize_name(user_name)
    user_dir = os.path.join(DATABASE_DIR, safe_name)
    os.makedirs(user_dir, exist_ok=True)

    start_time = time.time()
    count = capture_faces(user_dir, user_name, safe_name)
    capture_time = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"CADASTRO CONCLUÍDO PARA: {user_name}")
    print(f"Faces capturadas: {count}")
    print(f"Tempo total: {capture_time:.1f} segundos")
    print(f"Taxa de captura: {count / capture_time:.1f} faces/segundo")
    print(f"Armazenado em: {user_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()