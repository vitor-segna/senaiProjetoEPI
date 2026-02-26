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
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

nomes_conhecidos = {}
modelo_treinado = False
LIMITE_CONFIANCA_FACE = 60

# ==============================================================================
# 3. FUNÇÕES DE BANCO E TREINAMENTO
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
        face_img = cv2.resize(frame_gray[y:y+h, x:x+w], (200, 200))
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

print("\n>>> RECONHECIMENTO FACIAL ATIVO")
print(">>> 'c' para cadastrar novo aluno | 'q' para sair\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detectadas = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detectadas:
        if modo_cadastro:
            # LÓGICA DE CADASTRO
            if cadastro_count < 25:
                cadastro_count += 1
                salvar_nova_face(gray, x, y, w, h, cad_id, cad_nome)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
                cv2.putText(frame, f"Capturando {cadastro_count}/25", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                time.sleep(0.05)
            else:
                modo_cadastro = False
                treinar_modelo()
                cadastro_count = 0
                print(f"[SUCESSO] {cad_nome} cadastrado!")
        
        else:
            # LÓGICA DE RECONHECIMENTO
            id_nome = "Desconhecido"
            cor = (0, 0, 255) # Vermelho para desconhecido

            if modelo_treinado:
                roi_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                uid, confidencia = recognizer.predict(roi_gray)

                if confidencia < LIMITE_CONFIANCA_FACE:
                    id_nome = nomes_conhecidos.get(uid, f"ID {uid}")
                    cor = (0, 255, 0) # Verde para conhecido
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
            cv2.putText(frame, id_nome, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

    cv2.imshow("Cadastro e Reconhecimento Facial", frame)
    
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