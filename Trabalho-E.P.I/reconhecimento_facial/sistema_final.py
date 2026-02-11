import cv2
import mysql.connector
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import winsound  # Apenas para Windows

# --- BIBLIOTECAS PARA CADASTRO ---
import tkinter as tk
from tkinter import simpledialog

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',       # <--- SUA SENHA
    'database': 'epi_guard', 
    'port': 3308
}

EPI_OCULOS_ID = 1
EPI_CAPACETE_ID = 2

CLASSES_YOLO = [
    "hard hat", "helmet", "safety helmet",           # 0, 1, 2
    "person",                                        # 3
    "glasses", "sunglasses", "reading glasses",      # 4, 5, 6
    "safety goggles", "protective eyewear", "safety glasses" # 7, 8, 9
]

HELMET_CLASSES = [0, 1, 2] 
PERSON_CLASS = 3
ALL_EYEWEAR = [4, 5, 6, 7, 8, 9]

LIMITE_CONFIANCA = 60 

# ==============================================================================
# 2. INICIALIZAÇÃO
# ==============================================================================
print("[SISTEMA] Carregando Modelos...")
model = YOLO("yolov8s-worldv2.pt") 
model.set_classes(CLASSES_YOLO)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

nomes_conhecidos = {}
modelo_treinado = False

# ==============================================================================
# 3. BANCO E TREINO
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
        face_img = cv2.resize(frame_gray[y:y+h, x:x+w], (200, 200))
        _, buf = cv2.imencode('.jpg', face_img)
        cursor.execute("INSERT INTO amostras_facial (aluno_id, imagem) VALUES (%s, %s)", (uid, buf.tobytes()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERRO SALVAR] {e}")

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
            id1 = cursor.lastrowid
            cursor.execute("INSERT INTO evidencias (ocorrencia_id, imagem) VALUES (%s, %s)", (id1, nome_arquivo))
        if falta_oculos:
            cursor.execute("INSERT INTO ocorrencias (aluno_id, data_hora, epi_id) VALUES (%s, NOW(), %s)", (aluno_id, EPI_OCULOS_ID))
            id2 = cursor.lastrowid
            cursor.execute("INSERT INTO evidencias (ocorrencia_id, imagem) VALUES (%s, %s)", (id2, nome_arquivo))
        
        conn.commit()
        conn.close()
        print(f"[MULTA] Registrada para ID {aluno_id}.")
        threading.Thread(target=lambda: winsound.Beep(2500, 1000)).start()
    except Exception as e:
        print(f"[ERRO MULTA] {e}")

# ==============================================================================
# 4. FUNÇÕES DE ANÁLISE VISUAL (AJUSTADA PARA DETALHES PEQUENOS)
# ==============================================================================

def verificar_hsv_capacete(img_crop):
    if img_crop is None or img_crop.size == 0: return False
    h, w = img_crop.shape[:2]
    topo = img_crop[0:int(h*0.7), :] 
    hsv = cv2.cvtColor(topo, cv2.COLOR_BGR2HSV)
    mask_valid = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 255, 255]))
    ratio = cv2.countNonZero(mask_valid) / (topo.shape[0]*topo.shape[1])
    return ratio > 0.35

def verificar_cor_epi_oculos(img_crop):
    """
    DETECTA DETALHES PEQUENOS (HASTES AMARELAS / CENTRO VERMELHO).
    Removemos a limpeza de ruído agressiva e focamos em encontrar
    os pixels das cores específicas nas posições esperadas.
    """
    if img_crop is None or img_crop.size == 0: return False
    
    h, w = img_crop.shape[:2]
    
    # Analisa a metade superior onde ficam as hastes e o nariz
    roi = img_crop[0:int(h * 0.5), :]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # --- AJUSTE DE SENSIBILIDADE ---
    # Baixamos a Saturação Mínima para 100 para pegar amarelo em sombra ou fino
    sat_min = 100
    val_min = 80

    # 1. DEFINIR CORES
    # Amarelo (Hastes)
    lower_yellow = np.array([18, sat_min, val_min])
    upper_yellow = np.array([35, 255, 255])
    
    # Vermelho (Detalhe Central)
    lower_red1 = np.array([0, sat_min, val_min])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, sat_min, val_min])
    upper_red2 = np.array([180, 255, 255])

    # 2. CRIAR MÁSCARAS
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

    # --- TRUQUE: DILATAÇÃO ---
    # Como as hastes são finas, vamos "engrossar" o que achamos para facilitar a contagem
    # Em vez de apagar (erosão), vamos aumentar (dilatação)
    kernel = np.ones((3,3), np.uint8)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)
    mask_red = cv2.dilate(mask_red, kernel, iterations=1)

    # 3. CONTAGEM POR ZONA (OPCIONAL, MAS AJUDA)
    # Total de pixels na imagem recortada
    total_pixels = roi.shape[0] * roi.shape[1]
    
    count_yellow = cv2.countNonZero(mask_yellow)
    count_red = cv2.countNonZero(mask_red)
    
    ratio_yellow = count_yellow / total_pixels
    ratio_red = count_red / total_pixels

    # --- LÓGICA FINAL ---
    # Como são detalhes PEQUENOS, a porcentagem aceitável deve ser BAIXA.
    # Exigimos: 
    # - OU 2% de Amarelo (Hastes laterais)
    # - OU 1% de Vermelho (Detalhe no nariz)
    
    tem_detalhe_amarelo = ratio_yellow > 0.02
    tem_detalhe_vermelho = ratio_red > 0.01
    
    # Se tiver qualquer um dos dois detalhes vivos, é considerado EPI
    if tem_detalhe_amarelo or tem_detalhe_vermelho:
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

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if modo_cadastro:
        faces_rect = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if cadastro_count < 25:
                cadastro_count += 1
                salvar_nova_face(gray, x, y, w, h, cad_id, cad_nome)
                cv2.putText(frame, f"Capturando {cadastro_count}/25", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                time.sleep(0.1) 
            else:
                modo_cadastro = False
                treinar_modelo()
                cadastro_count = 0
                print("[SUCESSO] Cadastro finalizado!")
        cv2.imshow("EPI GUARD ULTIMATE", frame)

    else:
        results = model.predict(frame, conf=0.35, verbose=False, imgsz=640)
        
        pessoas_yolo = []
        capacetes = []
        oculos_detectados = []
        
        for r in results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                cls = int(box.cls[0])
                
                if cls == PERSON_CLASS: 
                    pessoas_yolo.append(coords)
                elif cls in HELMET_CLASSES: 
                    capacetes.append(coords)
                elif cls in ALL_EYEWEAR:
                    oculos_detectados.append(coords)

        frame_display = cv2.GaussianBlur(frame, (21, 21), 0)

        # Foca na maior pessoa
        pessoa_foco = None
        maior_area = 0
        for p in pessoas_yolo:
            area = (p[2]-p[0]) * (p[3]-p[1])
            if area > maior_area:
                maior_area = area
                pessoa_foco = p

        if pessoa_foco is not None:
            px1, py1, px2, py2 = pessoa_foco
            h_img, w_img = frame.shape[:2]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w_img, px2), min(h_img, py2)
            frame_display[py1:py2, px1:px2] = frame[py1:py2, px1:px2]

            roi_gray = gray[py1:py2, px1:px2]
            faces_haar = face_cascade.detectMultiScale(roi_gray, 1.1, 5)
            identidade_id = None
            identidade_nome = "Desconhecido"
            
            if len(faces_haar) > 0 and modelo_treinado:
                (fx, fy, fw, fh) = max(faces_haar, key=lambda b: b[2]*b[3])
                rosto_x, rosto_y = px1 + fx, py1 + fy
                try:
                    roi_face = cv2.resize(roi_gray[fy:fy+fh, fx:fx+fw], (200, 200))
                    uid, dist = recognizer.predict(roi_face)
                    if dist < LIMITE_CONFIANCA:
                        identidade_id = uid
                        identidade_nome = nomes_conhecidos.get(uid, f"ID {uid}")
                except: pass

            h_person = py2 - py1
            zona_cabeca = py1 + (h_person * 0.35)
            zona_olhos = py1 + (h_person * 0.55)
            
            tem_capacete = False
            tem_oculos = False
            
            # --- CAPACETE ---
            for (hx1, hy1, hx2, hy2) in capacetes:
                hcx = (hx1 + hx2) / 2
                if px1 < hcx < px2 and hy1 < zona_cabeca:
                    crop = frame[hy1:hy2, hx1:hx2] 
                    if verificar_hsv_capacete(crop):
                        tem_capacete = True
                        cv2.rectangle(frame_display, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                        cv2.putText(frame_display, "CAPACETE", (hx1, hy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # --- ÓCULOS ---
            for (ox1, oy1, ox2, oy2) in oculos_detectados:
                ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                
                if px1 < ocx < px2 and py1 < ocy < zona_olhos:
                    # Margem lateral maior (30%) para pegar bem as hastes
                    largura = ox2 - ox1
                    mx = int(largura * 0.3) 
                    crop_x1, crop_x2 = max(0, ox1 - mx), min(w_img, ox2 + mx)
                    crop_oculos = frame[oy1:oy2, crop_x1:crop_x2]
                    
                    if verificar_cor_epi_oculos(crop_oculos):
                        tem_oculos = True
                        cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                        cv2.putText(frame_display, "EPI DETALHES", (ox1, oy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    else:
                        cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                        cv2.putText(frame_display, "OCULOS COMUM", (ox1, oy2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            falha = not (tem_capacete and tem_oculos)
            status_texto = "EPI OK"
            cor_box = (0, 255, 0)

            if falha:
                cor_box = (0, 0, 255)
                status_texto = "INFRACAO"
                if not tem_capacete: status_texto += " [CAPACETE]"
                if not tem_oculos: status_texto += " [OCULOS]"

                if identidade_id:
                    if identidade_id not in tempo_infracao:
                        tempo_infracao[identidade_id] = time.time()
                    elif time.time() - tempo_infracao[identidade_id] > 3.0:
                        threading.Thread(target=registrar_multa, args=(frame.copy(), identidade_id, not tem_capacete, not tem_oculos)).start()
                        tempo_infracao[identidade_id] = time.time() + 10
            else:
                if identidade_id in tempo_infracao: del tempo_infracao[identidade_id]

            cv2.rectangle(frame_display, (px1, py1), (px2, py2), cor_box, 2)
            cv2.putText(frame_display, f"{identidade_nome}", (px1, py1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_box, 2)
            cv2.putText(frame_display, f"{status_texto}", (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_box, 2)

        cv2.imshow("EPI GUARD ULTIMATE", frame_display)
    
    k = cv2.waitKey(1)
    if k == ord('q'): break
    elif k == ord('c') and not modo_cadastro:
        root = tk.Tk()
        root.withdraw() 
        root.attributes('-topmost', True)
        cad_id = simpledialog.askinteger("Cadastro", "ID:", parent=root)
        if cad_id:
            cad_nome = simpledialog.askstring("Cadastro", "Nome:", parent=root)
            if cad_nome: modo_cadastro = True
        root.destroy()

cap.release()
cv2.destroyAllWindows()