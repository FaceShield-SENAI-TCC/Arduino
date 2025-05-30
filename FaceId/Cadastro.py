import cv2
import os
import time
import re
import numpy as np
from datetime import datetime

# ====================== CONFIGURAÇÕES ======================
DATABASE_DIR = "facial_database"  # Pasta principal do banco de dados
CAPTURE_DURATION = 10  # segundos (aumentado para 10 segundos)
TARGET_FACES = 50  # Queremos cerca de 50 fotos
HAAR_CASCADE = "haarcascade_frontalface_default.xml"
MIN_FACE_SIZE = (100, 100)  # Tamanho mínimo do rosto


# ====================== FUNÇÕES AUXILIARES ======================
def sanitize_name(name):
    """Remove caracteres especiais e espaços para criar nomes de arquivo seguros"""
    return re.sub(r'[^a-zA-Z0-9_]', '', name.replace(' ', '_'))


# ====================== FUNÇÃO PARA CADASTRAR USUÁRIO ======================
def register_user():
    """Solicita e valida o nome do usuário, cria pasta correspondente"""
    os.makedirs(DATABASE_DIR, exist_ok=True)

    while True:
        user_name = input("Digite o nome completo do usuário para cadastro: ").strip()
        if not user_name:
            print("Nome inválido. Tente novamente.")
            continue

        # Cria um nome seguro para pastas/arquivos
        safe_name = sanitize_name(user_name)
        user_dir = os.path.join(DATABASE_DIR, safe_name)

        if os.path.exists(user_dir):
            print(f"Usuário '{user_name}' já cadastrado. Deseja adicionar mais amostras? (s/n)")
            if input().lower() != 's':
                continue
        else:
            os.makedirs(user_dir)

        return user_name, safe_name, user_dir


# ====================== DETECÇÃO E CAPTURA FACIAL ======================
def capture_faces(user_dir, user_name, safe_name):
    """Captura rostos durante 10 segundos tentando obter cerca de 50 amostras"""
    # Carregar classificador Haar
    if not os.path.exists(HAAR_CASCADE):
        download_haarcascade()

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)

    # Iniciar captura de vídeo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Erro ao abrir a câmera!")

    print(f"\nPosicione seu rosto. Captura iniciando em 3 segundos...")
    time.sleep(3)

    start_time = time.time()
    last_capture = start_time
    face_count = len(os.listdir(user_dir))  # Contar amostras existentes

    print("Capturando... (Pressione ESC para encerrar antecipadamente)")

    # Calcular intervalo desejado entre capturas (0.2 segundos para 5 FPS)
    capture_interval = 0.2

    while (time.time() - start_time) < CAPTURE_DURATION:
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar contagem regressiva e nome do usuário
        elapsed = time.time() - start_time
        remaining = CAPTURE_DURATION - int(elapsed)
        display_text = f"{user_name} - Tempo: {remaining}s | Fotos: {face_count}/{TARGET_FACES}"
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detectar rostos em tempo real
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Tentar salvar a cada intervalo de capture_interval segundos
            current_time = time.time()
            if current_time - last_capture >= capture_interval:
                # Garantir que as coordenadas são válidas
                if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                    face_img = frame[y:y + h, x:x + w]

                    # Aplicar pré-processamento para melhorar qualidade
                    face_img = enhance_face_image(face_img)

                    face_count += 1
                    # Nome do arquivo com nome do usuário e número sequencial
                    face_filename = f"{safe_name}_{face_count:03d}.jpg"
                    face_path = os.path.join(user_dir, face_filename)
                    cv2.imwrite(face_path, face_img)
                    print(f"Rosto salvo: {face_filename}")
                    last_capture = current_time

                    # Parar se atingir o número desejado
                    if face_count >= TARGET_FACES:
                        break

        cv2.imshow('Cadastro Facial - Pressione ESC para sair', frame)
        key = cv2.waitKey(1)
        if key == 27:  # Tecla ESC
            break
        elif face_count >= TARGET_FACES:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nCadastro concluído! {face_count} amostras faciais salvas para {user_name}.")
    return face_count


def enhance_face_image(face_img):
    """Melhora a qualidade da imagem facial para melhor reconhecimento"""
    # Converter para escala de cinza
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Equalizar histograma para melhor contraste
    equalized = cv2.equalizeHist(gray)

    # Reduzir ruído com filtro bilateral
    denoised = cv2.bilateralFilter(equalized, 9, 75, 75)

    # Converter de volta para BGR
    enhanced = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    return enhanced


# ====================== PREPARAÇÃO PARA FACEID ======================
def prepare_for_faceid(safe_name):
    """Prepara os dados para uso no sistema de reconhecimento"""
    user_dir = os.path.join(DATABASE_DIR, safe_name)

    if not os.path.exists(user_dir) or len(os.listdir(user_dir)) == 0:
        print(f"Usuário não possui amostras cadastradas!")
        return

    print("\nPreparando dados para o sistema FaceID...")

    # Criar arquivo de metadados
    metadata_path = os.path.join(user_dir, "user_info.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Nome: {safe_name}\n")
        f.write(f"Data do Cadastro: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de Amostras: {len(os.listdir(user_dir)) - 1}\n")  # -1 para excluir o próprio metadata

    print(f"Estrutura de dados pronta para reconhecimento facial em: {user_dir}")

    print("\nPróximos passos para implementar o FaceID:")
    print("1. Desenvolver o sistema de reconhecimento usando a pasta 'facial_database'")
    print("2. Implementar a comparação de embeddings faciais")
    print("3. Configurar o acesso baseado na identificação facial")


# ====================== SUPORTE PARA HAAR CASCADE ======================
def download_haarcascade():
    """Baixa o classificador Haar Cascade se necessário"""
    import urllib.request
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    print("Baixando classificador Haar Cascade...")
    urllib.request.urlretrieve(url, HAAR_CASCADE)
    print("Download completo!")


# ====================== EXECUÇÃO PRINCIPAL ======================
if __name__ == "__main__":
    print("======== SISTEMA DE CADASTRO FACIAL ========")
    print("Este programa irá cadastrar seu rosto para uso em sistemas de segurança")
    print(f"Serão capturadas aproximadamente {TARGET_FACES} fotos em {CAPTURE_DURATION} segundos\n")

    user_name, safe_name, user_dir = register_user()
    capture_faces(user_dir, user_name, safe_name)
    prepare_for_faceid(safe_name)

    print("\nCadastro realizado com sucesso! Pressione Enter para sair.")
    input()