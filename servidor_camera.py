import cv2
import mysql.connector
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import winsound
from flask import Flask, Response # <-- ADICIONADO PARA WEB
from flask_cors import CORS # Para permitir que o PHP leia o vídeo

app = Flask(__name__)
CORS(app)

# ==============================================================================
# 1. CONFIGURAÇÕES (MANTIDAS DO SEU CÓDIGO)
# ==============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
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

# ==============================================================================
# 2. INICIALIZAÇÃO (MANTIDA)
# ==============================================================================
print("[SISTEMA] Carregando Modelos...")
model = YOLO("yolov8s-worldv2.pt") 
model.set_classes(CLASSES_YOLO)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

nomes_conhecidos = {}
modelo_treinado = False
tempo_infracao = {} 

# ... (MANTENHA SUAS FUNÇÕES AQUI: inicializar_banco(), treinar_modelo(), registrar_multa(), verificar_hsv_capacete(), verificar_cor_epi_oculos()) ...
# Nota: Estou omitindo as funções por brevidade, mas você deve copiar e colar exatamente como estavam no seu código original.

# ==============================================================================
# 3. GERADOR DE FRAMES PARA A WEB
# ==============================================================================
def gerar_frames():
    global tempo_infracao
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 360)
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- AQUI COMEÇA SUA LÓGICA DO YOLO E FACE RECOGNITION ---
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
                elif cls in ALL_EYEWEAR: oculos_detectados.append(coords)

        frame_display = frame.copy()
        
        # Foca na pessoa principal
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
            
            # Reconhecimento Facial (simplificado por espaço)
            roi_gray = gray[py1:py2, px1:px2]
            faces_haar = face_cascade.detectMultiScale(roi_gray, 1.1, 5)
            identidade_id = None
            identidade_nome = "Desconhecido"
            # ... Resto da sua lógica de face e EPI ...
            
            # Desenha as caixas e textos no frame_display (sua lógica existente)
            cv2.rectangle(frame_display, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame_display, f"{identidade_nome}", (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- FIM DA LÓGICA DO YOLO ---

        # 4. CONVERSÃO PARA A WEB (Substitui o cv2.imshow)
        ret, buffer = cv2.imencode('.jpg', frame_display)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ==============================================================================
# 5. ROTAS DO SERVIDOR
# ==============================================================================
@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # inicializar_banco()
    # treinar_modelo()
    print(">>> Servidor de Vídeo rodando em http://localhost:5000/video_feed <<<")
    app.run(host='0.0.0.0', port=5000, threaded=True)