import cv2
import mysql.connector
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import winsound  # Apenas para Windows

# --- BIBLIOTECAS PARA AS JANELAS DE CADASTRO ---
import tkinter as tk
from tkinter import simpledialog

# ==============================================================================
# 1. CONFIGURAÇÕES GERAIS
# ==============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',       # <--- SUA SENHA
    'database': 'epi_guard', 
    'port': 3308          # <--- SUA PORTA (3306 ou 3308)
}

# IDs dos EPIs no Banco de Dados
EPI_OCULOS_ID = 1
EPI_CAPACETE_ID = 2

# --- CONFIGURAÇÃO DAS CLASSES DO YOLO ---
CLASSES_YOLO = [
    "hard hat", "helmet", "safety helmet",  # 0, 1, 2 (Capacetes)
    "person",                               # 3 (Pessoa)
    "glasses", "safety glasses", "goggles", "eye protection" # 4, 5, 6, 7 (Óculos)
]

HELMET_CLASSES = [0, 1, 2] 
PERSON_CLASS = 3
GLASSES_CLASSES = [4, 5, 6, 7]

# Ajuste de Reconhecimento Facial
LIMITE_CONFIANCA = 60  # (Menor = mais rigoroso)

# ==============================================================================
# 2. INICIALIZAÇÃO DOS MODELOS
# ==============================================================================
print("[SISTEMA] Carregando IA YOLO e Reconhecimento Facial...")
model = YOLO("yolov8s-worldv2.pt")
model.set_classes(CLASSES_YOLO)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

nomes_conhecidos = {}
modelo_treinado = False

# ==============================================================================
# 3. FUNÇÕES DE BANCO DE DADOS E TREINAMENTO
# ==============================================================================
def inicializar_banco():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS alunos (id INT PRIMARY KEY, nome VARCHAR(100))")
        cursor.execute("CREATE TABLE IF NOT EXISTS amostras_facial (id INT AUTO_INCREMENT PRIMARY KEY, aluno_id INT, imagem LONGBLOB)")
        cursor.execute("CREATE TABLE IF NOT EXISTS ocorrencias (id INT AUTO_INCREMENT PRIMARY KEY, aluno_id INT, data_hora DATETIME, epi_id INT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS evidencias (id INT AUTO_INCREMENT PRIMARY KEY, ocorrencia_id INT, imagem VARCHAR(255))")
        cursor.execute("CREATE TABLE IF NOT EXISTS epis (id INT PRIMARY KEY, nome VARCHAR(50))")
        cursor.execute("INSERT IGNORE INTO epis (id, nome) VALUES (1, 'Oculos'), (2, 'Capacete')")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERRO BD] {e}")

def treinar_modelo():
    global modelo_treinado, nomes_conhecidos
    print("[TREINO] Lendo banco de dados para aprender rostos...")
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
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                ids.append(uid)
        
        if len(faces) > 0:
            recognizer.train(faces, np.array(ids))
            modelo_treinado = True
            print(f"[TREINO] {len(faces)} faces aprendidas. Sistema pronto!")
        else:
            modelo_treinado = False
            print("[TREINO] Nenhuma face cadastrada ainda.")
        conn.close()
    except Exception as e:
        print(f"[ERRO TREINO] {e}")

def salvar_nova_face(frame_gray, x, y, w, h, uid, nome):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO alunos (id, nome) VALUES (%s, %s) ON DUPLICATE KEY UPDATE nome=%s", (uid, nome, nome))
    face_img = cv2.resize(frame_gray[y:y+h, x:x+w], (200, 200))
    _, buf = cv2.imencode('.jpg', face_img)
    cursor.execute("INSERT INTO amostras_facial (aluno_id, imagem) VALUES (%s, %s)", (uid, buf.tobytes()))
    conn.commit()
    conn.close()

def registrar_multa(frame_evidencia, aluno_id, falta_capacete, falta_oculos):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"multa_{aluno_id}_{timestamp}.jpg"
        if not os.path.exists("evidencias"): os.makedirs("evidencias")
        cv2.imwrite(f"evidencias/{nome_arquivo}", frame_evidencia)
        
        if falta_capacete:
            cursor.execute("INSERT INTO ocorrencias (aluno_id, data_hora, epi_id) VALUES (%s, NOW(), %s)", (aluno_id, EPI_CAPACETE_ID))
            id_oc1 = cursor.lastrowid
            cursor.execute("INSERT INTO evidencias (ocorrencia_id, imagem) VALUES (%s, %s)", (id_oc1, nome_arquivo))

        if falta_oculos:
            cursor.execute("INSERT INTO ocorrencias (aluno_id, data_hora, epi_id) VALUES (%s, NOW(), %s)", (aluno_id, EPI_OCULOS_ID))
            id_oc2 = cursor.lastrowid
            cursor.execute("INSERT INTO evidencias (ocorrencia_id, imagem) VALUES (%s, %s)", (id_oc2, nome_arquivo))
        
        conn.commit()
        conn.close()
        print(f"[MULTA] Aluno {aluno_id} registrado.")
        threading.Thread(target=lambda: winsound.Beep(2500, 1000)).start()
    except Exception as e:
        print(f"[ERRO BD] {e}")

# ==============================================================================
# 4. FUNÇÕES DE ANÁLISE VISUAL (COR)
# ==============================================================================

def verificar_hsv_capacete(img_crop):
    """ Verifica se o capacete tem uma cor válida """
    if img_crop is None or img_crop.size == 0: return False
    h, w = img_crop.shape[:2]
    topo = img_crop[0:int(h*0.7), :] 
    hsv = cv2.cvtColor(topo, cv2.COLOR_BGR2HSV)
    mask_valid = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 255, 255]))
    ratio = cv2.countNonZero(mask_valid) / (topo.shape[0]*topo.shape[1])
    return ratio > 0.35

def verificar_cor_epi_oculos(img_crop):
    """
    Verifica se o óculos tem detalhes AMARELOS (hastes) ou VERMELHOS (frente).
    Retorna True se for EPI, False se for óculos comum.
    """
    if img_crop is None or img_crop.size == 0: return False
    
    # Converter para HSV
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # --- CORES DO EPI ---
    
    # 1. AMARELO (Hastes) - Range de Amarelo
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 2. VERMELHO (Detalhe Frontal) - Range duplo no HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    # Junta as duas máscaras (Amarelo OU Vermelho)
    mask_final = cv2.add(mask_yellow, mask_red)
    
    # Calcula porcentagem de cor na imagem do óculos
    pixels_coloridos = cv2.countNonZero(mask_final)
    total_pixels = img_crop.shape[0] * img_crop.shape[1]
    
    ratio = pixels_coloridos / total_pixels
    
    # Se mais de 2% da área do recorte tiver essas cores, é o EPI correto
    # (Valor baixo pois são apenas DETALHES nas hastes/frente)
    if ratio > 0.02: 
        return True
    
    return False

# ==============================================================================
# 5. LOOP PRINCIPAL
# ==============================================================================
inicializar_banco()
treinar_modelo()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

modo_cadastro = False
cadastro_count = 0
cad_id, cad_nome = 0, ""
tempo_infracao = {} 

print("\n>>> SISTEMA PRONTO <<<")
print("Tecle 'C' para abrir a janela de Cadastro | 'Q' para Sair")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ---------------------------------------------------------
    # MODO CADASTRO
    # ---------------------------------------------------------
    if modo_cadastro:
        faces_rect = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if cadastro_count < 25:
                cadastro_count += 1
                salvar_nova_face(gray, x, y, w, h, cad_id, cad_nome)
                cv2.putText(frame, f"Capturando {cadastro_count}/25", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                time.sleep(0.05)
            else:
                modo_cadastro = False
                treinar_modelo()
                cadastro_count = 0
                print("[SUCESSO] Cadastro finalizado e modelo retreinado!")
        
        cv2.imshow("EPI GUARD ULTIMATE", frame)

    # ---------------------------------------------------------
    # MODO VIGILÂNCIA
    # ---------------------------------------------------------
    else:
        # Detecta objetos
        results = model.predict(frame, conf=0.35, verbose=False, imgsz=640)
        
        pessoas_yolo = []
        capacetes = []
        oculos_detectados = []
        
        for r in results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                cls = int(box.cls[0])
                if cls == PERSON_CLASS: pessoas_yolo.append(coords)
                elif cls in HELMET_CLASSES: capacetes.append(coords)
                elif cls in GLASSES_CLASSES: oculos_detectados.append(coords)

        # Fundo borrado
        frame_display = cv2.GaussianBlur(frame, (51, 51), 0)

        # Encontra a pessoa principal
        pessoa_foco = None
        maior_area = 0

        for p in pessoas_yolo:
            w_p = p[2] - p[0]
            h_p = p[3] - p[1]
            area = w_p * h_p
            if area > maior_area:
                maior_area = area
                pessoa_foco = p

        if pessoa_foco is not None:
            px1, py1, px2, py2 = pessoa_foco
            
            h_img, w_img = frame.shape[:2]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w_img, px2), min(h_img, py2)
            
            frame_display[py1:py2, px1:px2] = frame[py1:py2, px1:px2]

            # --- ANÁLISE DE SEGURANÇA ---
            cor_box = (0, 255, 0) # Verde
            
            # Identificação Facial
            roi_gray = gray[py1:py2, px1:px2]
            faces_haar = face_cascade.detectMultiScale(roi_gray, 1.1, 5)
            
            identidade_id = None
            identidade_nome = "Desconhecido"
            
            if len(faces_haar) > 0 and modelo_treinado:
                (fx, fy, fw, fh) = max(faces_haar, key=lambda b: b[2]*b[3])
                rosto_global_x = px1 + fx
                rosto_global_y = py1 + fy
                
                cv2.rectangle(frame_display, (rosto_global_x, rosto_global_y), (rosto_global_x+fw, rosto_global_y+fh), (255, 255, 0), 1)
                
                try:
                    roi_face = cv2.resize(roi_gray[fy:fy+fh, fx:fx+fw], (200, 200))
                    uid_pred, dist = recognizer.predict(roi_face)
                    if dist < LIMITE_CONFIANCA:
                        identidade_id = uid_pred
                        identidade_nome = nomes_conhecidos.get(uid_pred, f"ID {uid_pred}")
                except: pass

            # Verificação de EPI
            h_person = py2 - py1
            zona_cabeca = py1 + (h_person * 0.3)
            zona_olhos = py1 + (h_person * 0.5)
            
            tem_capacete = False
            tem_oculos = False
            
            # 1. CAPACETE
            for (hx1, hy1, hx2, hy2) in capacetes:
                hcx = (hx1 + hx2) / 2
                if px1 < hcx < px2 and hy1 < zona_cabeca:
                    crop = frame[hy1:hy2, hx1:hx2] 
                    if verificar_hsv_capacete(crop):
                        tem_capacete = True
                        cv2.rectangle(frame_display, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
            
            # 2. ÓCULOS (COM VERIFICAÇÃO DE DETALHES AMARELOS/VERMELHOS)
            for (ox1, oy1, ox2, oy2) in oculos_detectados:
                ocx = (ox1 + ox2) / 2
                ocy = (oy1 + oy2) / 2
                
                if px1 < ocx < px2 and oy1 < ocy < zona_olhos:
                    # Expande o recorte horizontalmente para pegar as hastes laterais
                    largura_oculos = ox2 - ox1
                    margem_x = int(largura_oculos * 0.2) # 20% de margem
                    
                    crop_x1 = max(0, ox1 - margem_x)
                    crop_x2 = min(w_img, ox2 + margem_x)
                    
                    # Recorta o óculos (com margem)
                    crop_oculos = frame[oy1:oy2, crop_x1:crop_x2]
                    
                    # Verifica se tem amarelo ou vermelho
                    if verificar_cor_epi_oculos(crop_oculos):
                        tem_oculos = True
                        cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                        cv2.putText(frame_display, "EPI OK", (ox1, oy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    else:
                        # Detectou óculos, mas não tem as cores do EPI -> Óculos comum
                        cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (255, 0, 0), 1)
                        cv2.putText(frame_display, "Pessoal", (ox1, oy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

            # Lógica de Infração
            falha = not (tem_capacete and tem_oculos)
            status_texto = "EPI OK"

            if falha:
                cor_box = (0, 0, 255) # Vermelho
                status_texto = "SEM EPI"
                
                if identidade_id is not None:
                    status_texto += f" ({identidade_nome})"
                    if identidade_id not in tempo_infracao:
                        tempo_infracao[identidade_id] = time.time()
                    elif time.time() - tempo_infracao[identidade_id] > 3.0:
                        threading.Thread(target=registrar_multa, args=(frame.copy(), identidade_id, not tem_capacete, not tem_oculos)).start()
                        tempo_infracao[identidade_id] = time.time() + 10
                else:
                    if identidade_id in tempo_infracao: del tempo_infracao[identidade_id]
            else:
                if identidade_id in tempo_infracao: del tempo_infracao[identidade_id]

            cv2.rectangle(frame_display, (px1, py1), (px2, py2), cor_box, 2)
            cv2.putText(frame_display, f"{identidade_nome} | {status_texto}", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_box, 2)
            cv2.putText(frame_display, "[ALVO FOCADO]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("EPI GUARD ULTIMATE", frame_display)
    
    k = cv2.waitKey(1)
    if k == ord('q'): 
        break
    
    elif k == ord('c') and not modo_cadastro:
        root = tk.Tk()
        root.withdraw() 
        root.attributes('-topmost', True)
        cad_id = simpledialog.askinteger("Novo Cadastro", "Digite o ID do Usuário (Número):", parent=root)
        if cad_id is not None:
            cad_nome = simpledialog.askstring("Novo Cadastro", "Digite o NOME do Usuário:", parent=root)
            if cad_nome:
                modo_cadastro = True
                print(f"[CADASTRO] Iniciando captura para: {cad_nome} (ID: {cad_id})")
        root.destroy()

cap.release()
cv2.destroyAllWindows()