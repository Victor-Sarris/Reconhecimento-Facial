import requests
import os

url = "http://192.168.18.149:5000/api/cadastrar_direto"
pasta_imagens = "dataset/PedroLucas"
nome_cliente = "Pedro Lucas Alves de Assis Cardoso"

if not os.path.exists(pasta_imagens):
    print(f" Erro: A pasta '{pasta_imagens}' não foi encontrada.")
    exit()

arquivos = os.listdir(pasta_imagens)
fotos_validas = [f for f in arquivos if f.lower().endswith((".jpg", ".jpeg", ".png"))]
total = len(fotos_validas)

print(f"--- Iniciando envio em massa para o Labrador/Totem de Reconhecimento---")
print(f"Pasta: {pasta_imagens}")
print(f"Total de fotos encontradas: {total}\n")
print(f"Nome do Paciente: {nome_cliente}\n")

sucessos = 0

for i, arquivo in enumerate(fotos_validas):
    caminho_completo = os.path.join(pasta_imagens, arquivo)

    print(f"[{i+1}/{total}] Enviando {arquivo}...", end=" ")

    try:
        with open(caminho_completo, "rb") as f:
            files = {"foto": f}
            data = {"nome": nome_cliente}

            response = requests.post(url, files=files, data=data)

        if response.status_code == 201:
            print("Sucesso!")
            sucessos += 1
        else:
            print(f"Falha: {response.json().get('erro')}")

    except Exception as e:
        print(f"Erro de conexão: {e}")

print(f"\n--- CONCLUÍDO! ---")
print(f" {sucessos} fotos processadas e salvas no banco de dados do Labrador.")
