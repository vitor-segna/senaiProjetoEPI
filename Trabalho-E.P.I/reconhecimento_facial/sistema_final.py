import cv2
import mysql.connector
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import simpledialog

# ==============================================================================
# 1. CONFIGURAÇÕES DO BANCO DE DADOS
# ==============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '', 
    'database': 'epi_guard', 
    'port': 3308
}

# ==============================================================================
# 2. INICIALIZAÇÃO DOS MODELOS DE FACE
# ==============================================================================
print("[SISTEMA] Carregando Modelos de Face...")
# NOVIDADE: Carregando tanto o rosto de frente quanto o de perfil!
cascade_frente = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cascade_perfil = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

nomes_conhecidos = {}
modelo_treinado = False
LIMITE_CONFIANCA_FACE = 60

# ==============================================================================
# 3. FUNÇÕES DE BANCO E TREINAMENTO (Mantido Intacto)
# ==============================================================================
def inicializar_banco():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS alunos (id INT PRIMARY KEY, nome VARCHAR(100))")
        cursor.execute("CREATE TABLE IF NOT EXISTS amostras_facial (id INT AUTO_INCREMENT PRIMARY KEY, aluno_id INT, imagem LONGBLOB)")
        conn.commit()
        conn.close()
        print("[BD] Banco de dados pronto.")
    except Exception as e:
        print(f"[ERRO BD] {e}")

def treinar_modelo():
    global modelo_treinado, nomes_conhecidos
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT id, nome FROM alunos")
        nomes_conhecidos = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT aluno_id, imagem FROM amostras_facial")
        faces, ids = [], []
        for uid, blob in cursor.fetchall():
            if blob:
                nparr = np.frombuffer(blob, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(cv2.resize(img, (200, 200)))
                    ids.append(uid)
        
        if len(faces) > 0:
            recognizer.train(faces, np.array(ids))
            modelo_treinado = True
            print(f"[TREINO] Reconhecimento facial pronto. {len(faces)} amostras carregadas.")
        else:
            modelo_treinado = False
        conn.close()
    except Exception as e:
        print(f"[ERRO TREINO] {e}")

def salvar_nova_face(frame_gray, x, y, w, h, uid, nome):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO alunos (id, nome) VALUES (%s, %s) ON DUPLICATE KEY UPDATE nome=%s", (uid, nome, nome))
        
        h_corte = int(h * 0.60)
        face_img = cv2.resize(frame_gray[y:y+h_corte, x:x+w], (200, 200))
        
        _, buf = cv2.imencode('.jpg', face_img)
        cursor.execute("INSERT INTO amostras_facial (aluno_id, imagem) VALUES (%s, %s)", (uid, buf.tobytes()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERRO SALVAR] {e}")

# ==============================================================================
# 4. LOOP PRINCIPAL
# ==============================================================================
inicializar_banco()
treinar_modelo()

cap = cv2.VideoCapture(0)
modo_cadastro = False
cadastro_count = 0
cad_id, cad_nome = 0, ""

# Variáveis para a "Memória Visual"
ultima_face = None
frames_sem_rosto = 0

print("\n>>> RECONHECIMENTO FACIAL ATIVO (FOCO 360 + DESFOQUE NATIVO)")
print(">>> 'c' para cadastrar novo aluno | 'q' para sair\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h_frame, w_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- 1. DETECÇÃO 360 GRAUS ---
    todas_caixas = []
    
    # Detecta rosto de frente
    faces_frente = cascade_frente.detectMultiScale(gray, 1.3, 5)
    for caixa in faces_frente: todas_caixas.append(caixa)
        
    # Detecta rosto de perfil (virado para um lado)
    faces_perfil = cascade_perfil.detectMultiScale(gray, 1.3, 5)
    for caixa in faces_perfil: todas_caixas.append(caixa)
        
    # Detecta rosto de perfil (virado para o outro lado espelhando a imagem)
    gray_espelhado = cv2.flip(gray, 1)
    faces_perfil_esq = cascade_perfil.detectMultiScale(gray_espelhado, 1.3, 5)
    for (x, y, w, h) in faces_perfil_esq:
        # Desfaz o espelhamento para pegar a coordenada real na tela
        x_real = w_frame - x - w
        todas_caixas.append((x_real, y, w, h))

    # --- 2. ESCOLHER O ROSTO PRINCIPAL E USAR A MEMÓRIA ---
    rosto_principal = None
    maior_area = 0
    
    # Pega apenas a maior caixa detectada (para evitar bugar se detectar frente e perfil junto)
    for (x, y, w, h) in todas_caixas:
        area = w * h
        if area > maior_area:
            maior_area = area
            rosto_principal = (x, y, w, h)
            
    # Sistema de Memória: Se perder o rosto, lembra onde estava por 15 frames
    if rosto_principal is not None:
        ultima_face = rosto_principal
        frames_sem_rosto = 0
    else:
        frames_sem_rosto += 1
        if frames_sem_rosto < 15 and ultima_face is not None:
            rosto_principal = ultima_face # Usa a memória para não desfocar
        else:
            ultima_face = None # Esquece de vez após muito tempo

    # --- 3. EFEITO DE DESFOQUE ---
    fundo_desfocado = cv2.GaussianBlur(frame, (71, 71), 0)
    mask = np.zeros_like(gray)

    if rosto_principal is not None:
        (x, y, w, h) = rosto_principal
        centro = (x + w // 2, y + h // 2)
        eixos = (int(w * 0.9), int(h * 1.3)) 
        cv2.ellipse(mask, centro, eixos, 0, 0, 360, 255, -1)

    mask_suave = cv2.GaussianBlur(mask, (51, 51), 0)
    mask_3d = cv2.cvtColor(mask_suave, cv2.COLOR_GRAY2BGR) / 255.0

    frame_final = (frame * mask_3d + fundo_desfocado * (1 - mask_3d)).astype(np.uint8)

    # --- 4. APLICAÇÃO DE RECONHECIMENTO ---
    if rosto_principal is not None:
        (x, y, w, h) = rosto_principal
        h_corte = int(h * 0.60)
        
        if modo_cadastro:
            if cadastro_count < 25:
                cadastro_count += 1
                salvar_nova_face(gray, x, y, w, h, cad_id, cad_nome)
                
                cv2.rectangle(frame_final, (x, y), (x+w, y+h), (255, 165, 0), 1)
                cv2.line(frame_final, (x, y+h_corte), (x+w, y+h_corte), (255, 165, 0), 2)
                
                cv2.putText(frame_final, f"Capturando {cadastro_count}/25", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                time.sleep(0.05)
            else:
                modo_cadastro = False
                treinar_modelo()
                cadastro_count = 0
                print(f"[SUCESSO] {cad_nome} cadastrado com foco superior!")
        
        else:
            # OBS: O reconhecimento funciona melhor de frente. De lado ele pode dar "Desconhecido", 
            # mas o DESFOQUE (foco) agora vai continuar em você graças à lógica 360!
            id_nome = "Analisando..."
            cor = (0, 0, 255) 

            if modelo_treinado:
                # Usa a imagem cinza original (nítida) para o IA analisar
                roi_gray = cv2.resize(gray[y:y+h_corte, x:x+w], (200, 200))
                uid, confidencia = recognizer.predict(roi_gray)

                if confidencia < LIMITE_CONFIANCA_FACE:
                    id_nome = nomes_conhecidos.get(uid, f"ID {uid}")
                    cor = (0, 255, 0) 
            
            cv2.rectangle(frame_final, (x, y), (x+w, y+h), cor, 1)
            cv2.line(frame_final, (x, y+h_corte), (x+w, y+h_corte), cor, 2) 
            cv2.putText(frame_final, id_nome, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

    cv2.imshow("Cadastro e Reconhecimento Facial", frame_final)
    
    key = cv2.waitKey(1)
    if key == ord('q'): 
        break
    elif key == ord('c') and not modo_cadastro:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        cid = simpledialog.askinteger("Cadastro", "Digite o ID (Número):", parent=root)
        if cid:
            cnome = simpledialog.askstring("Cadastro", "Digite o Nome:", parent=root)
            if cnome:
                cad_id, cad_nome = cid, cnome
                modo_cadastro = True
        root.destroy()

cap.release()
cv2.destroyAllWindows()