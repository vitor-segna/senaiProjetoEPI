import cv2
import numpy as np
from ultralytics import YOLO
import time
import winsound  # Apenas Windows
import threading 

# --- CONFIGURAÇÃO INICIAL ---
print("Carregando modelo YOLO...")
model = YOLO("yolov8s-worldv2.pt")

model.set_classes([
    "hard hat", "helmet", "safety helmet", 
    "person", 
    "glasses", "eye protection", "goggles"
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

# ==============================================================================
# 1. FUNÇÃO DE VALIDAÇÃO DE ÓCULOS
# ==============================================================================
def verificar_oculos_epi(img_crop):
    if img_crop is None or img_crop.size == 0: return False, None

    h, w = img_crop.shape[:2]
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Amarelo e Vermelho (Detalhes do EPI)
    lower_yellow = np.array([20, 135, 120]) 
    upper_yellow = np.array([35, 255, 255])
    
    lower_red1, upper_red1 = np.array([0, 150, 90]), np.array([8, 255, 255])
    lower_red2, upper_red2 = np.array([172, 150, 90]), np.array([180, 255, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # EROSÃO: Remove linhas finas
    kernel = np.ones((3,3), np.uint8)
    mask_yellow = cv2.erode(mask_yellow, kernel, iterations=1)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=2) 

    # Análise de Zonas
    largura_haste = int(w * 0.25)
    roi_esq = mask_yellow[:, :largura_haste]
    roi_dir = mask_yellow[:, w-largura_haste:]
    roi_centro_red = mask_red[:, largura_haste:w-largura_haste]
    
    pixels_amarelo = cv2.countNonZero(roi_esq) + cv2.countNonZero(roi_dir)
    pixels_vermelho = cv2.countNonZero(roi_centro_red)
    
    area_hastes = (h * largura_haste) * 2
    area_centro = h * (w - 2*largura_haste)

    is_valid = False
    if (pixels_amarelo / area_hastes) > 0.020: is_valid = True
    elif (pixels_vermelho / area_centro) > 0.025: is_valid = True

    return is_valid, mask_yellow

# ==============================================================================
# 2. FUNÇÃO DE VALIDAÇÃO DE CAPACETE CINZA
# ==============================================================================
def verificar_capacete_cinza(img_crop):
    if img_crop is None or img_crop.size == 0: return False, None

    h, w = img_crop.shape[:2]
    # Analisa apenas os 70% superiores do recorte
    crop_topo = img_crop[0:int(h*0.7), :] 
    
    hsv = cv2.cvtColor(crop_topo, cv2.COLOR_BGR2HSV)

    # --- DEFINIÇÃO DE CINZA EM HSV ---
    # Saturação (S): 0-50 (MUITO BAIXA)
    lower_gray = np.array([0, 0, 60])
    upper_gray = np.array([180, 50, 230])

    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    # --- MORFOLOGIA ---
    kernel = np.ones((5,5), np.uint8) 
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)

    # Contagem
    total_pixels = crop_topo.shape[0] * crop_topo.shape[1]
    gray_pixels = cv2.countNonZero(mask_gray)
    
    ratio = gray_pixels / total_pixels

    # Se mais de 40% da área superior for cinza sólido
    is_valid = ratio > 0.40

    return is_valid, mask_gray

# --- LOOP PRINCIPAL ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Sistema Iniciado: Modo EPI COMPLETO.")

while True:
    success, img_original = cap.read()
    if not success: break

    img_display = img_original.copy()
    
    debug_oculos = np.zeros((100,100), dtype=np.uint8)
    debug_capacete = np.zeros((100,100), dtype=np.uint8)

    results = model.predict(img_original, conf=0.3, imgsz=640, verbose=False)

    persons = []
    capacetes_detectados = [] 
    oculos_detectados = []    
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            coords = list(map(int, box.xyxy[0]))
            x1, y1, x2, y2 = coords
            
            if cls == PERSON_ID:
                persons.append(coords)
            
            elif cls in HELMET_IDS:
                y1_c, y2_c = max(0, y1), min(img_original.shape[0], y2)
                x1_c, x2_c = max(0, x1), min(img_original.shape[1], x2)
                helmet_crop = img_original[y1_c:y2_c, x1_c:x2_c]
                
                eh_cinza, mask_h = verificar_capacete_cinza(helmet_crop)
                capacetes_detectados.append((coords, eh_cinza))
                if mask_h is not None and eh_cinza: debug_capacete = mask_h

            elif cls in GLASSES_IDS:
                y1_c, y2_c = max(0, y1), min(img_original.shape[0], y2)
                x1_c, x2_c = max(0, x1), min(img_original.shape[1], x2)
                glasses_crop = img_original[y1_c:y2_c, x1_c:x2_c]
                
                eh_epi, mask_g = verificar_oculos_epi(glasses_crop)
                oculos_detectados.append((coords, eh_epi))
                if mask_g is not None and eh_epi: debug_oculos = mask_g

    # --- LÓGICA DE ASSOCIAÇÃO ---
    main_person = None
    if persons:
        main_person = max(persons, key=lambda p: (p[2]-p[0]) * (p[3]-p[1]))
        px1, py1, px2, py2 = main_person
        
        # Desenha a pessoa (Borrando o fundo)
        img_blurred = cv2.GaussianBlur(img_original, (21, 21), 0)
        img_display = img_blurred.copy()
        img_display[py1:py2, px1:px2] = img_original[py1:py2, px1:px2] 

        h_p = py2 - py1
        zona_cabeca_topo = py1 + (h_p * 0.25)
        zona_olhos = py1 + (h_p * 0.50)      

        status_capacete = "FALTA" 
        status_oculos = "FALTA"   

        # 1. VERIFICA CAPACETES
        for (hx1, hy1, hx2, hy2), eh_cinza in capacetes_detectados:
            hcx = (hx1 + hx2) / 2
            if px1 < hcx < px2 and hy1 < zona_cabeca_topo:
                if eh_cinza:
                    status_capacete = "OK"
                    cv2.rectangle(img_display, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                    cv2.putText(img_display, "CAPACETE CINZA", (hx1, hy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    break 
                else:
                    status_capacete = "ERRADO"
                    cv2.rectangle(img_display, (hx1, hy1), (hx2, hy2), (0, 165, 255), 2) # Laranja
                    # --- MUDANÇA AQUI ---
                    cv2.putText(img_display, "EQUIPAMENTO INVALIDO", (hx1, hy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

        # 2. VERIFICA ÓCULOS
        for (gx1, gy1, gx2, gy2), eh_valido in oculos_detectados:
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2
            
            if px1 < gcx < px2 and gcy < zona_olhos:
                if eh_valido:
                    status_oculos = "OK"
                    cv2.rectangle(img_display, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cv2.putText(img_display, "EPI OLHOS", (gx1, gy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    break
                elif status_oculos == "FALTA": 
                    status_oculos = "ERRADO"
                    cv2.rectangle(img_display, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                    cv2.putText(img_display, "MODELO NAO PERMITIDO", (gx1, gy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # --- DECISÃO FINAL ---
        seguro = (status_capacete == "OK" and status_oculos == "OK")
        cor_borda = (0, 255, 0) if seguro else (0, 0, 255)
        
        cv2.rectangle(img_display, (px1, py1), (px2, py2), cor_borda, 3)
        info_text = f"CAPACETE: {status_capacete} | OCULOS: {status_oculos}"
        cv2.putText(img_display, info_text, (px1, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_borda, 2)

        if not seguro:
            if time.time() - ultimo_aviso > INTERVALO_AVISO:
                tocar_alarme()
                ultimo_aviso = time.time()

    else:
        img_display = cv2.GaussianBlur(img_original, (21, 21), 0)
        cv2.putText(img_display, "AGUARDANDO PESSOA...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Detector EPI Completo", img_display)
    cv2.imshow("Filtro Capacete (Cinza)", cv2.resize(debug_capacete, (200, 150)))
    cv2.imshow("Filtro Oculos (Cor)", cv2.resize(debug_oculos, (200, 150)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()