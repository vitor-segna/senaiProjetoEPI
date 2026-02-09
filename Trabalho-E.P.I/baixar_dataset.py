from roboflow import Roboflow

# Substitua pela sua chave que você pegou no site do Roboflow
api_key = "SUA_CHAVE_AQUI" 

rf = Roboflow(api_key=api_key)

# Aqui estamos conectando a um dataset público de EPI (PPE)
# Este ID "vamsi-q79ie/construction-site-safety-f70v1" é um exemplo real e popular
project = rf.workspace("vamsi-q79ie").project("construction-site-safety-f70v1")

# Escolhe a versão do dataset (geralmente a mais recente)
dataset = project.version(1).download("yolov8")

print(f"Dataset baixado com sucesso em: {dataset.location}") 