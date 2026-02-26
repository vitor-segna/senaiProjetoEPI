import cv2
import mysql.connector
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import winsound # Apenas para Windows
from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',       # Confirme sua senha
    'database': 'epi_guard', 
    'port': 3308
}

EPI_OCULOS_ID = 1
EPI_CAPACETE_ID = 2

CLASSES_YOLO = [
    "hard hat", "helmet", "safety helmet", 
    "person", 
    "glasses", "sunglasses", "reading glasses", 
    "safety goggles", "protective eyewear", "safety glasses"
]
HELMET_CLASSES = [0, 1, 2] 
PERSON_CLASS = 3
ALL_EYEWEAR = [4, 5, 6, 7, 8, 9]
LIMITE_CONFIANCA_FACE = 60

# Variáveis Globais
nomes_conhecidos = {}
modelo_treinado = False
tempo_infracao = {}

# ==============================================================================
# 2. INICIALIZAÇÃO DOS MODELOS
# ==============================================================================
print("[SISTEMA] Carregando Modelos...")
model = YOLO("yolov8s-worldv2.pt") 
model.set_classes(CLASSES_YOLO)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ==============================================================================
# 3. FUNÇÕES DE BANCO DE DADOS E LÓGICA DE EPI
# ==============================================================================
def inicializar_banco():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS alunos (id INT PRIMARY KEY, nome VARCHAR(100))")
        cursor.execute("CREATE TABLE IF NOT EXISTS amostras_facial (id INT AUTO_INCREMENT PRIMARY KEY, aluno_id INT, imagem LONGBLOB)")
        cursor.execute("CREATE TABLE IF NOT EXISTS ocorrencias (id INT AUTO_INCREMENT PRIMARY KEY, aluno_id INT, data_hora DATETIME, epi_id INT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS evidencias (id INT AUTO_INCREMENT PRIMARY KEY, ocorrencia_id INT, imagem LONGBLOB)")
        cursor.execute("CREATE TABLE IF NOT EXISTS epis (id INT PRIMARY KEY, nome VARCHAR(50))")
        cursor.execute("INSERT IGNORE INTO epis (id, nome) VALUES (1, 'Oculos'), (2, 'Capacete')")
        conn.commit()
        conn.close()
        print("[BD] Banco inicializado com sucesso.")
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
            print(f"[TREINO] Modelo facial treinado com {len(faces)} faces.")
        else:
            modelo_treinado = False
            print("[TREINO] Nenhuma face cadastrada ainda.")
        conn.close()
    except Exception as e:
        print(f"[ERRO TREINO] {e}")

def registrar_multa(frame_evidencia, aluno_id, falta_capacete, falta_oculos):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        _, buffer = cv2.imencode('.jpg', frame_evidencia)
        imagem_bytes = buffer.tobytes()

        if falta_capacete:
            cursor.execute("INSERT INTO ocorrencias (aluno_id, data_hora, epi_id) VALUES (%s, NOW(), %s)", (aluno_id, EPI_CAPACETE_ID))
            id_last = cursor.lastrowid
            cursor.execute("INSERT INTO evidencias (ocorrencia_id, imagem) VALUES (%s, %s)", (id_last, imagem_bytes))

        if falta_oculos:
            cursor.execute("INSERT INTO ocorrencias (aluno_id, data_hora, epi_id) VALUES (%s, NOW(), %s)", (aluno_id, EPI_OCULOS_ID))
            id_last = cursor.lastrowid
            cursor.execute("INSERT INTO evidencias (ocorrencia_id, imagem) VALUES (%s, %s)", (id_last, imagem_bytes))

        conn.commit()
        conn.close()
        print(f"[MULTA] Registrada no banco para o ID {aluno_id}.")
        threading.Thread(target=lambda: winsound.Beep(2500, 1000)).start()
    except Exception as e:
        print(f"[ERRO MULTA] {e}")

def verificar_hsv_capacete(img_crop):
    if img_crop is None or img_crop.size == 0: return False
    h, w = img_crop.shape[:2]
    topo = img_crop[0:int(h*0.7), :] 
    hsv = cv2.cvtColor(topo, cv2.COLOR_BGR2HSV)
    mask_valid = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 255, 255]))
    ratio = cv2.countNonZero(mask_valid) / (topo.shape[0]*topo.shape[1])
    return ratio > 0.35

def verificar_cor_epi_oculos(img_crop):
    if img_crop is None or img_crop.size == 0: return False
    
    # Padronização
    img_crop = cv2.resize(img_crop, (200, 90))
    img_crop = cv2.GaussianBlur(img_crop, (5,5), 0)
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    h, w = img_crop.shape[:2]

    # --- 1. FILTRO DE COR AMARELA (Hastes) ---
    # Saturação mínima subiu de 80 para 130 (Isso mata o óculos preto/reflexo)
    # Valor (brilho) mínimo subiu para 100
    lower_yellow = np.array([18, 130, 100]) 
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # --- 2. FILTRO DE COR VERMELHA (Centro) ---
    # Saturação mínima subiu para 140 (Ignora tons de pele e reflexos)
    lower_red1 = np.array([0, 140, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 140, 80])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # --- 3. FILTRO ANTI-PRETO/TRANSPARENTE (Segurança extra) ---
    # Captura o que é muito escuro (Preto) ou muito sem cor (Transparente/Cinza)
    # Se a imagem for quase toda assim, cancelamos a detecção
    mask_neutra = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 70, 255]))
    pixels_neutros = cv2.countNonZero(mask_neutra)

    # --- 4. LÓGICA DE ZONAS ---
    largura_lateral = int(w * 0.35) 
    zona_esq = mask_yellow[:, :largura_lateral]
    zona_dir = mask_yellow[:, w - largura_lateral:]
    zona_centro = mask_red[:, int(w*0.25):int(w*0.75)]

    cnt_esq = cv2.countNonZero(zona_esq)
    cnt_dir = cv2.countNonZero(zona_dir)
    cnt_centro = cv2.countNonZero(zona_centro)

    # --- 5. CRITÉRIO DE DECISÃO ---
    # Se mais de 80% da área do óculos for "cor neutra" (preto/transparente), 
    # ignoramos mesmo que haja um pequeno reflexo.
    if pixels_neutros > (img_crop.size * 0.85):
        return False

    # Precisamos de uma "massa" mínima de cor vibrante (mais de 40 pixels)
    tem_vermelho = cnt_centro > 30
    tem_amarelo = (cnt_esq > 35 or cnt_dir > 35)

    if tem_vermelho or tem_amarelo:
        return True
        
    return False
# ==============================================================================
# 4. GERADOR DE FRAMES COM LÓGICA APLICADA
# ==============================================================================
def gerar_frames():
    global tempo_infracao, modelo_treinado, nomes_conhecidos
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model.predict(frame, conf=0.35, verbose=False, imgsz=640)
        
        pessoas_yolo, capacetes, oculos_detectados = [], [], []
        
        for r in results:
            for box in r.boxes:
                coords = list(map(int, box.xyxy[0]))
                cls = int(box.cls[0])
                if cls == PERSON_CLASS: pessoas_yolo.append(coords)
                elif cls in HELMET_CLASSES: capacetes.append(coords)
                elif cls in ALL_EYEWEAR: oculos_detectados.append(coords)

        frame_display = frame.copy()
        
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
            
            roi_gray = gray[py1:py2, px1:px2]
            faces_haar = face_cascade.detectMultiScale(roi_gray, 1.1, 5)
            identidade_id = None
            identidade_nome = "Desconhecido"
            
            if len(faces_haar) > 0 and modelo_treinado:
                (fx, fy, fw, fh) = max(faces_haar, key=lambda b: b[2]*b[3])
                try:
                    roi_face = cv2.resize(roi_gray[fy:fy+fh, fx:fx+fw], (200, 200))
                    uid, dist = recognizer.predict(roi_face)
                    if dist < LIMITE_CONFIANCA_FACE:
                        identidade_id = uid
                        identidade_nome = nomes_conhecidos.get(uid, f"ID {uid}")
                except: pass

            h_person = py2 - py1
            zona_cabeca = py1 + (h_person * 0.35)
            zona_olhos = py1 + (h_person * 0.55)
            
            tem_capacete, tem_oculos = False, False
            
            # Checagem Capacete
            for (hx1, hy1, hx2, hy2) in capacetes:
                hcx = (hx1 + hx2) / 2
                if px1 < hcx < px2 and hy1 < zona_cabeca:
                    if verificar_hsv_capacete(frame[hy1:hy2, hx1:hx2]):
                        tem_capacete = True
                        cv2.rectangle(frame_display, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)

            # Checagem Óculos
            for (ox1, oy1, ox2, oy2) in oculos_detectados:
                ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                if px1 < ocx < px2 and py1 < ocy < zona_olhos:
                    largura = ox2 - ox1
                    margem = int(largura * 0.5) 
                    crop_x1 = max(0, ox1 - margem)
                    crop_x2 = min(w_img, ox2 + margem)
                    crop_oculos = frame[oy1:oy2, crop_x1:crop_x2]
                    
                    if verificar_cor_epi_oculos(crop_oculos):
                        tem_oculos = True
                        cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                        cv2.putText(frame_display, "EPI OK", (ox1, oy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    else:
                        cv2.rectangle(frame_display, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                        cv2.putText(frame_display, "COMUM", (ox1, oy2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # Lógica de Infração e Banco de Dados
            falha = not (tem_capacete and tem_oculos)
            cor_box = (0, 255, 0)
            status_texto = "APROVADO"

            if falha:
                cor_box = (0, 0, 255)
                status_texto = "INFRACAO"
                if not tem_capacete: status_texto += " [CAPACETE]"
                if not tem_oculos: status_texto += " [OCULOS]"
                
                if identidade_id:
                    agora = time.time()
                    if identidade_id not in tempo_infracao:
                        tempo_infracao[identidade_id] = agora
                    # Aguarda 3 segundos de infração contínua para multar
                    elif agora - tempo_infracao[identidade_id] > 3.0:
                        threading.Thread(target=registrar_multa, args=(frame.copy(), identidade_id, not tem_capacete, not tem_oculos)).start()
                        tempo_infracao[identidade_id] = agora + 10 # Dá 10 segundos de carência antes da próxima
            else:
                if identidade_id in tempo_infracao: del tempo_infracao[identidade_id]

            cv2.rectangle(frame_display, (px1, py1), (px2, py2), cor_box, 2)
            cv2.putText(frame_display, f"{identidade_nome} | {status_texto}", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_box, 2)

        ret, buffer = cv2.imencode('.jpg', frame_display)
        frame_bytes = buffer.tobytes()
        
        # Adicione este bloco TRY/EXCEPT para capturar quando o PHP cortar o vídeo
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            # O navegador fechou a conexão (você apertou o botão vermelho)
            print("Conexão de vídeo encerrada pelo painel.")
            break # Isso quebra o loop "while True"
        except Exception as e:
            print(f"Erro na transmissão: {e}")
            break

    # Ao quebrar o loop, a câmera é finalmente liberada (apaga a luzinha)
    cap.release()

# ==============================================================================
# 5. ROTAS DO SERVIDOR
# ==============================================================================
@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Descomentado para preparar o ambiente antes de rodar o servidor Web
    inicializar_banco()
    treinar_modelo()
    print(">>> Servidor Web de Visão Computacional rodando em http://localhost:5000/video_feed <<<")
    app.run(host='0.0.0.0', port=5000, threaded=True)