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

# --- FUNÇÕES DE COR (REJEIÇÃO DE ÓCULOS COMUNS) ---
def verificar_detalhes_epi(img_crop):
    if img_crop is None or img_crop.size == 0: return False, None, None

    h, w = img_crop.shape[:2]
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # --- 1. DEFINIÇÃO DE CORES (MAIS RIGOROSA) ---
    # Amarelo:
    # H (Matiz): 20-35 (Cortei o 15-20 pois confunde com pele alaranjada)
    # S (Saturação): 135+ (Só cores MUITO vivas. Metal/Pele tem saturação baixa)
    # V (Brilho): 120+ (Ignora marrom/amarelo escuro)
    lower_yellow = np.array([20, 135, 120]) 
    upper_yellow = np.array([35, 255, 255])
    
    # Vermelho (Detalhe Central)
    lower_red1, upper_red1 = np.array([0, 150, 90]), np.array([8, 255, 255])
    lower_red2, upper_red2 = np.array([172, 150, 90]), np.array([180, 255, 255])

    # --- 2. CRIAR MÁSCARAS ---
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # --- 3. EROSÃO (A TÉCNICA CHAVE) ---
    # Isso apaga linhas finas (armações de metal/grau).
    # Só sobra o que for GROSSO (hastes de plástico).
    kernel_erosao = np.ones((3,3), np.uint8)
    mask_yellow = cv2.erode(mask_yellow, kernel_erosao, iterations=1)
    
    # Depois da erosão, dilatamos um pouco para conectar o que sobrou
    mask_yellow = cv2.dilate(mask_yellow, kernel_erosao, iterations=2)
    mask_red = cv2.dilate(mask_red, kernel_erosao, iterations=2)

    # --- 4. ANÁLISE POR ZONAS ---
    largura_haste = int(w * 0.25) # Olhar 25% das bordas
    roi_esq = mask_yellow[:, :largura_haste]
    roi_dir = mask_yellow[:, w-largura_haste:]
    
    roi_centro_red = mask_red[:, largura_haste:w-largura_haste]
    
    pixels_amarelo = cv2.countNonZero(roi_esq) + cv2.countNonZero(roi_dir)
    pixels_vermelho = cv2.countNonZero(roi_centro_red)
    
    area_hastes = (h * largura_haste) * 2
    area_centro = h * (w - 2*largura_haste)

    # --- 5. VALIDAÇÃO RIGOROSA ---
    is_valid = False

    # Aumentei a exigência de pixels para 2% da área lateral
    # Como fizemos erosão, se sobrar 2%, é porque era algo grosso e amarelo.
    if (pixels_amarelo / area_hastes) > 0.020: 
        is_valid = True
    
    # Vermelho no centro (ponte do nariz ou elástico)
    elif (pixels_vermelho / area_centro) > 0.025:
        is_valid = True

    return is_valid, mask_yellow, mask_red

# --- LOOP PRINCIPAL ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Sistema Iniciado: Modo REJEIÇÃO DE ÓCULOS FINOS.")

while True:
    success, img_original = cap.read()
    if not success: break

    img_display = img_original.copy()
    
    debug_yellow = np.zeros((100,100), dtype=np.uint8)
    debug_red = np.zeros((100,100), dtype=np.uint8)

    results = model.predict(img_original, conf=0.3, imgsz=640, verbose=False)

    persons = []
    helmets = []
    epis_olhos_validos = [] 
    oculos_invalidos = [] 
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            coords = list(map(int, box.xyxy[0]))
            x1, y1, x2, y2 = coords
            
            if cls == PERSON_ID:
                persons.append(coords)
            elif cls in HELMET_IDS:
                helmets.append(coords)
            elif cls in GLASSES_IDS:
                # Recorte com margem de segurança (para pegar bem a haste)
                y1_c, y2_c = max(0, y1), min(img_original.shape[0], y2)
                x1_c, x2_c = max(0, x1), min(img_original.shape[1], x2)
                glasses_crop = img_original[y1_c:y2_c, x1_c:x2_c]
                
                eh_valido, m_y, m_r = verificar_detalhes_epi(glasses_crop)
                
                if m_y is not None: debug_yellow = m_y
                if m_r is not None: debug_red = m_r

                if eh_valido:
                    epis_olhos_validos.append(coords)
                else:
                    oculos_invalidos.append(coords)

    # Foco na Pessoa
    main_person = None
    if persons:
        main_person = max(persons, key=lambda p: (p[2]-p[0]) * (p[3]-p[1]))
        img_blurred = cv2.GaussianBlur(img_original, (21, 21), 0)
        px1, py1, px2, py2 = main_person
        py1, py2 = max(0, py1), min(img_display.shape[0], py2)
        px1, px2 = max(0, px1), min(img_display.shape[1], px2)
        img_display = img_blurred
        img_display[py1:py2, px1:px2] = img_original[py1:py2, px1:px2]
    else:
        img_display = cv2.GaussianBlur(img_original, (21, 21), 0)

    # Decisão
    alguma_infracao = False
    
    if main_person:
        px1, py1, px2, py2 = main_person
        h_p = py2 - py1
        zona_olhos_base = py1 + (h_p * 0.60)
        
        tem_capacete = False
        tem_epi_olho = False
        tem_oculos_comum = False
        
        # Capacete
        for hx1, hy1, hx2, hy2 in helmets:
            hcx = (hx1+hx2)/2
            if (px1-50) <= hcx <= (px2+50):
                tem_capacete = True
                cv2.rectangle(img_display, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)

        # EPI (Específico)
        for ex1, ey1, ex2, ey2 in epis_olhos_validos:
            ecx, ecy = (ex1+ex2)/2, (ey1+ey2)/2
            if px1 <= ecx <= px2 and ecy <= zona_olhos_base:
                tem_epi_olho = True
                cv2.rectangle(img_display, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
                cv2.putText(img_display, "EPI AUTENTICO", (ex1, ey1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                break

        # Óculos Comum (Detectado pelo YOLO mas rejeitado pela cor)
        if not tem_epi_olho:
            for ix1, iy1, ix2, iy2 in oculos_invalidos:
                icx, icy = (ix1+ix2)/2, (iy1+iy2)/2
                if px1 <= icx <= px2 and icy <= zona_olhos_base:
                    tem_oculos_comum = True
                    cv2.rectangle(img_display, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)
                    cv2.putText(img_display, "MODELO NAO PERMITIDO", (ix1, iy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # Status
        if tem_capacete and tem_epi_olho:
            status = "LIBERADO"
            cor = (0, 255, 0)
        else:
            alguma_infracao = True
            cor = (0, 0, 255)
            if not tem_capacete: status = "SEM CAPACETE"
            elif tem_oculos_comum: status = "OCULOS PROIBIDO"
            else: status = "SEM OCULOS"
            
        cv2.rectangle(img_display, (px1, py1), (px2, py2), cor, 3)
        cv2.putText(img_display, status, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 3)

    if alguma_infracao:
        if time.time() - ultimo_aviso > INTERVALO_AVISO:
            tocar_alarme()
            ultimo_aviso = time.time()

    cv2.imshow("Detector", img_display)
    
    # Observe estas janelas:
    # Óculos COMUM deve ficar TUDO PRETO (a erosão apaga as linhas finas)
    # EPI deve mostrar MANCHAS BRANCAS GORDAS
    cv2.imshow("Debug Amarelo", cv2.resize(debug_yellow, (200, 100)))
    cv2.imshow("Debug Vermelho", cv2.resize(debug_red, (200, 100)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()