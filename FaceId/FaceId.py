import cv2
import os
import numpy as np
import time
from deepface import DeepFace
from datetime import datetime

# ====================== CONFIGURAÇÕES ======================
DATABASE_DIR = "facial_database"  # Mesma pasta usada no cadastro
MODEL_NAME = "VGG-Face"  # Modelo de reconhecimento
DISTANCE_THRESHOLD = 0.55  # Limiar para considerar reconhecimento válido
MIN_FACE_SIZE = (100, 100)  # Tamanho mínimo do rosto
RECOGNITION_COOLDOWN = 1.5  # Segundos entre verificações


# ====================== CARREGAR BANCO DE DADOS FACIAL ======================
def load_facial_database():
    print("Carregando banco de dados facial...")
    database = {}

    for user_folder in os.listdir(DATABASE_DIR):
        user_path = os.path.join(DATABASE_DIR, user_folder)

        if os.path.isdir(user_path):
            # Extrair nome real do usuário
            user_name = " ".join([part.capitalize() for part in user_folder.split('_')])
            embeddings = []

            for face_file in os.listdir(user_path):
                if face_file.endswith((".jpg", ".png", ".jpeg")):
                    face_path = os.path.join(user_path, face_file)

                    try:
                        # Gerar embedding para cada rosto cadastrado
                        embedding_obj = DeepFace.represent(
                            img_path=face_path,
                            model_name=MODEL_NAME,
                            detector_backend="opencv",  # Usar OpenCV para evitar problemas
                            enforce_detection=False
                        )
                        embeddings.append(embedding_obj[0]["embedding"])
                    except Exception as e:
                        print(f"Erro ao processar {face_path}: {str(e)}")

            if embeddings:
                database[user_name] = embeddings
                print(f"» {user_name}: {len(embeddings)} amostras carregadas")

    if not database:
        print("Banco de dados vazio! Cadastre usuários primeiro.")

    return database


# ====================== FUNÇÃO DE RECONHECIMENTO ======================
def recognize_face(face_img, database):
    """Compara um rosto com o banco de dados e retorna a melhor correspondência"""
    try:
        # Gerar embedding para o rosto capturado
        captured_embedding_obj = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=False
        )
        captured_embedding = captured_embedding_obj[0]["embedding"]

        best_match = None
        min_distance = float('inf')

        # Comparar com todos os embeddings do banco
        for user_name, embeddings in database.items():
            for db_embedding in embeddings:
                # Calcular distância Euclidiana
                distance = np.linalg.norm(np.array(captured_embedding) - np.array(db_embedding))

                if distance < min_distance and distance < DISTANCE_THRESHOLD:
                    min_distance = distance
                    best_match = user_name

        return best_match, min_distance

    except Exception as e:
        print(f"Erro no reconhecimento: {str(e)}")
        return None, None


# ====================== SISTEMA PRINCIPAL DE SEGURANÇA ======================
def facial_recognition_system():
    # Carregar banco de dados
    facial_db = load_facial_database()
    if not facial_db:
        return

    # Iniciar captura de vídeo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir câmera!")
        return

    print("\nSistema de reconhecimento facial ativado. Pressione ESC para sair.")

    # Carregar detector facial OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    last_recognition = 0  # Timestamp do último reconhecimento
    current_user = None
    confidence = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espelhar o frame para melhor experiência
        frame = cv2.flip(frame, 1)

        # Detectar rostos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )

        recognition_status = "Procurando..."
        color = (0, 165, 255)  # Laranja

        for (x, y, w, h) in faces:
            # Desenhar retângulo ao redor do rosto
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Verificar cooldown
            current_time = time.time()
            if current_time - last_recognition > RECOGNITION_COOLDOWN:
                # Recortar rosto
                face_img = frame[y:y + h, x:x + w]

                # Fazer reconhecimento
                current_user, confidence = recognize_face(face_img, facial_db)
                last_recognition = current_time

                if current_user:
                    recognition_status = f"Bem-vindo(a), {current_user}!"
                    color = (0, 255, 0)  # Verde
                else:
                    recognition_status = "Desconhecido"
                    color = (0, 0, 255)  # Vermelho
            elif current_user:
                recognition_status = f"Bem-vindo(a), {current_user}!"
                color = (0, 255, 0)  # Verde

        # Exibir status
        cv2.putText(frame, recognition_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Exibir confiança se reconhecido
        if current_user and confidence:
            cv2.putText(frame, f"Confianca: {1 - confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Exibir instruções
        cv2.putText(frame, "Pressione ESC para sair", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Sistema de Segurança Facial', frame)

        # Sair com ESC
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSistema encerrado.")


# ====================== EXECUÇÃO ======================
if __name__ == "__main__":
    facial_recognition_system()