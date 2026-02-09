import requests
import zipfile
import os

def download_and_extract():
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
    zip_path = "construction-ppe.zip"
    extract_dir = "dataset_epi"

    print(f"Baixando dataset de: {url}...")
    
    # Download do arquivo
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Download concluído. Extraindo arquivos...")
    
    # Extração
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Limpeza do arquivo zip
    os.remove(zip_path)
    
    print(f"Pronto! O dataset foi extraído na pasta: {os.path.abspath(extract_dir)}")

if __name__ == "__main__":
    download_and_extract()
