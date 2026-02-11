import cv2
import numpy as np
from ultralytics import YOLO
import time
import winsound  # Apenas para Windows
import threading 

# --- CONFIGURAÇÃO INICIAL ---
print("Carregando modelo YOLO...")
model = YOLO("yolov8s-worldv2.pt")

# Vocabulário expandido (melhora detecção)
model.set_classes([
    "hard hat", "helmet", "safety helmet",  # IDs 0, 1, 2
    "person",                               # ID 3
    "glasses", "eye protection", "goggles"  # IDs 4, 5, 6
])

HELMET_IDS = [0, 1, 2]
PERSON_ID = 3
GLASSES_IDS = [4, 5, 6]

# --- CONFIGURAÇÃO DE SOM ---
ultimo_aviso = 0
INTERVALO_AVISO = 2  

def tocar_alarme():
    def _beep():
        try:
            winsound.Beep(1000, 500)
        except:
            pass 
    t = threading.Thread(target=_beep)
    t.start()

# --- FUNÇÕES DE DETECÇÃO DE COR (Mantidas) ---
def verificar_amarelo_haste(img_crop):
    if img_crop is None or img_crop.size == 0: return False, None
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 70, 70]) 
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5,5), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    match_found = False
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > (img_crop.shape[0]*img_crop.shape[1] * 0.005): 
            match_found = True
    return match_found, mask_dilated

def verificar_vermelho_centro(img_crop):
    if img_crop is None or img_crop.size == 0: return False, None
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 140, 80]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 140, 80]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    if (cv2.countNonZero(mask) / (img_crop.shape[0]*img_crop.shape[1])) > 0.025: 
        return True, mask
    return False, mask

# --- LOOP PRINCIPAL ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Sistema Iniciado: MODO FOCO (Fundo Desfocado).")

while True:
    success, img_original = cap.read() # Lemos como img_original
    if not success: break

    # Cria uma cópia para ser a imagem final exibida
    img_display = img_original.copy()

    results = model.predict(img_original, conf=0.3, imgsz=640, verbose=False)

    persons = []
    helmets = []
    epis_olhos_validos = [] 
    
    # 1. COLETA DE DADOS
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            coords = list(map(int, box.xyxy[0]))
            x1, y1, x2, y2 = coords
            conf = float(box.conf[0])
            
            if cls == PERSON_ID:
                persons.append(coords)
            
            elif cls in HELMET_IDS:
                helmets.append(coords)
            
            elif cls in GLASSES_IDS:
                # Validação de óculos
                y1_c, y2_c = max(0, y1), min(img_original.shape[0], y2)
                x1_c, x2_c = max(0, x1), min(img_original.shape[1], x2)
                glasses_crop = img_original[y1_c:y2_c, x1_c:x2_c]
                
                tem_haste, _ = verificar_amarelo_haste(glasses_crop)
                tem_centro, _ = verificar_vermelho_centro(glasses_crop)
                
                if (tem_haste or tem_centro) or (conf > 0.60):
                    epis_olhos_validos.append(coords)

    # 2. LÓGICA DE FOCO E DESFOQUE
    main_person = None
    
    if persons:
        # Encontra a pessoa mais próxima (Maior área: largura * altura)
        main_person = max(persons, key=lambda p: (p[2]-p[0]) * (p[3]-p[1]))
        
        # PASSO A: Borrar toda a imagem de fundo
        # (21, 21) é a intensidade do desfoque. Deve ser número ímpar.
        img_blurred = cv2.GaussianBlur(img_original, (21, 21), 0)
        
        # PASSO B: Recortar a pessoa nítida da original e colar no fundo borrado
        px1, py1, px2, py2 = main_person
        
        # Ajuste de segurança para não sair da tela
        py1, py2 = max(0, py1), min(img_display.shape[0], py2)
        px1, px2 = max(0, px1), min(img_display.shape[1], px2)

        # A imagem exibida vira a borrada
        img_display = img_blurred
        
        # "Colamos" a pessoa nítida de volta
        img_display[py1:py2, px1:px2] = img_original[py1:py2, px1:px2]

    else:
        # Se não tem ninguém, mostra tudo borrado ou tudo normal (escolhi tudo borrado)
        img_display = cv2.GaussianBlur(img_original, (21, 21), 0)

    # 3. VALIDAÇÃO E DESENHO (Apenas na Pessoa Principal)
    alguma_infracao = False

    if main_person:
        px1, py1, px2, py2 = main_person
        altura_pessoa = py2 - py1
        
        # Verifica Capacete
        zona_capacete_topo = py1 - (altura_pessoa * 0.20)
        zona_capacete_base = py1 + (altura_pessoa * 0.40)
        tem_capacete = False
        
        for hx1, hy1, hx2, hy2 in helmets:
            hcx, hcy = (hx1+hx2)/2, (hy1+hy2)/2
            # Desenha capacete se estiver perto da pessoa principal
            if (px1 - 50) <= hcx <= (px2 + 50):
                cv2.rectangle(img_display, (hx1, hy1), (hx2, hy2), (0, 165, 255), 2)
                if zona_capacete_topo <= hcy <= zona_capacete_base:
                    tem_capacete = True

        # Verifica Óculos
        tem_epi_olho = False
        for ex1, ey1, ex2, ey2 in epis_olhos_validos:
            ecx, ecy = (ex1+ex2)/2, (ey1+ey2)/2
            if px1 <= ecx <= px2 and py1 <= ecy <= (py1 + altura_pessoa * 0.6):
                tem_epi_olho = True
                cv2.rectangle(img_display, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
                break

        # Status Final
        if tem_capacete and tem_epi_olho:
            status = "ACESSO LIBERADO"
            cor = (0, 255, 0)
        else:
            alguma_infracao = True
            cor = (0, 0, 255)
            if not tem_capacete and not tem_epi_olho: status = "SEM EPIS"
            elif not tem_capacete: 
                status = "SEM CAPACETE"
                cor = (0, 165, 255)
            elif not tem_epi_olho: status = "SEM OCULOS"
        
        # Desenha a caixa apenas na pessoa focada
        cv2.rectangle(img_display, (px1, py1), (px2, py2), cor, 3)
        cv2.putText(img_display, status, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor, 3)

    # --- ALARME ---
    if alguma_infracao:
        agora = time.time()
        if agora - ultimo_aviso > INTERVALO_AVISO:
            tocar_alarme()
            ultimo_aviso = agora

    cv2.imshow("Detector EPI - Efeito Foco", img_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()