import cv2
import face_recognition
import pickle
import numpy as np
import threading
import time
import requests
import os
import sqlite3
import math
import sys
import socket

from datetime import datetime
from flask import Flask, Response, jsonify, request
from cryptography.fernet import Fernet

# CONFIGURAÇÕES PRINCIPAIS
ARQUIVO_DADOS = "encodings.pickle"
BANCO_DADOS = "totem_banco.db"
URL_CAMERA = "http://192.168.1.40:4747/video"
INTERVALO_SCAN_IA = 4.0
# Adicione junto às outras variáveis globais partilhadas da IA
ultimo_nome_reconhecido = None  # Guarda o nome da última pessoa que gerou um log válido
DELAY_RECONHECIMENTO = 5.0

LARGURA_TELA = 1024
ALTURA_TELA = 600

COR_BARRA_FUNDO = (180, 0, 0)
COR_BTN_FUNDO = (20, 20, 20)
COR_TEXTO = (255, 255, 255)
COR_RECONHECIDO = (0, 255, 0)

MODO_RECONHECIMENTO = 0
MODO_CAPTURANDO = 1
MODO_INFO_REMOTO = 2

estado_atual = MODO_RECONHECIMENTO

app = Flask(__name__)
lock = threading.Lock()
frame_atual = None
lista_encodings = []
lista_nomes = []
nome_novo_cadastro = ""
buffer_fotos_novas = []

# VARIÁVEIS GLOBAIS PARTILHADAS PARA A THREAD DA IA
ia_processando = False
caixas_detectadas = []
nomes_detectados = []
ultimo_sucesso = 0
nome_detectado = ""

# MODULO A LASER (VL53L0X)
DISTANCIA_GATILHO_MM = 800  # 80 centímetros
pessoa_na_frente = True

# O Python busca a chave direto da memória do Sistema Operacional (Windows ou Linux)
CHAVE_SESSAO = os.environ.get("CHAVE_BIOMETRIA")

if not CHAVE_SESSAO:
    print("[ERRO CRÍTICO LGPD] Variável de ambiente CHAVE_BIOMETRIA não encontrada no sistema!")
    print("O Totem não pode iniciar sem a chave de segurança. Encerrando...")
    sys.exit(1) # Interrompe o programa por segurança

# Como a variável de ambiente vem como texto (string), usamos .encode() para transformar em bytes
fernet_cipher = Fernet(CHAVE_SESSAO.encode())

ARQUIVO_MATRIZ = "matriz_projecao.npy"

def carregar_ou_gerar_matriz_ortogonal(dimensao=128):
    """
    Gera uma matriz ortogonal secreta para aplicar o Bio-hashing.
    Preserva a distância euclidiana, essencial para o face_recognition.
    """
    if os.path.exists(ARQUIVO_MATRIZ):
        return np.load(ARQUIVO_MATRIZ)
    else:
        # 1. Cria uma matriz aleatória 128x128
        H = np.random.randn(dimensao, dimensao)
        # 2. Aplica a decomposição QR para extrair a matriz ortogonal (Q)
        Q, R = np.linalg.qr(H)
        
        # 3. Salva a matriz no disco (No futuro da clínica, isso virá da API)
        np.save(ARQUIVO_MATRIZ, Q)
        print("[SEGURANÇA LGPD] Nova Matriz Ortogonal de Bio-hashing gerada!")
        return Q

# Carrega a matriz para a memória RAM
MATRIZ_PROJECAO = carregar_ou_gerar_matriz_ortogonal(128)

def obter_ip_local():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# CRIAÇÃO DO BANCO DE DADOS
def iniciar_banco():
    conn = sqlite3.connect(BANCO_DADOS)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS Usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT UNIQUE,
            data_cadastro DATETIME,
            nivel_acesso TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS Logs_Acesso (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            data_hora DATETIME,
            confianca_reconhecimento REAL,
            foto_momento TEXT,
            FOREIGN KEY(usuario_id) REFERENCES Usuarios(id)
        )
    """)

    try:
        c.execute("ALTER TABLE Logs_Acesso ADD COLUMN status_acesso TEXT DEFAULT 'LIBERADO'")
        c.execute("ALTER TABLE Logs_Acesso ADD COLUMN tempo_inferencia_ms INTEGER DEFAULT 0")
        c.execute("ALTER TABLE Logs_Acesso ADD COLUMN hardware_temp_c REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()
    print("[BANCO] Banco de Dados Inicializado com Sucesso.")


def cadastrar_usuario_db(nome, nivel="Aluno"):
    conn = sqlite3.connect(BANCO_DADOS)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO Usuarios (nome, data_cadastro, nivel_acesso) VALUES (?, ?, ?)",
            (nome, datetime.now(), nivel),
        )
        conn.commit()
        print(f"[BANCO] Usuário '{nome}' registrado no banco.")
    except sqlite3.IntegrityError:
        print(f"[BANCO] Usuário '{nome}' já existe no banco.")
    finally:
        conn.close()


def registrar_acesso_db(nome, confianca, frame_capturado, tempo_ms):
    conn = sqlite3.connect(BANCO_DADOS)
    c = conn.cursor()

    c.execute("SELECT id FROM Usuarios WHERE nome = ?", (nome,))
    row = c.fetchone()

    if not row:
        c.execute(
            "INSERT INTO Usuarios (nome, data_cadastro, nivel_acesso) VALUES (?, ?, ?)",
            (nome, datetime.now(), "Migrado do Sistema Antigo"),
        )
        conn.commit()
        user_id = c.lastrowid
    else:
        user_id = row[0]

    agora_dt = datetime.now()

    c.execute(
        """
        INSERT INTO Logs_Acesso (usuario_id, data_hora, confianca_reconhecimento, foto_momento, status_acesso, tempo_inferencia_ms)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (user_id, agora_dt, confianca, "FOTO_DESCARTADA", "LIBERADO", tempo_ms),
    )

    conn.commit()
    print(f"[AUDITORIA] Acesso salvo: {nome} | Confiança: {confianca}% | Tempo: {tempo_ms}ms")
    conn.close()


# VÍDEO STREAM
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ultimo_frame = None
        self.rodando = False
        self.lock = threading.Lock()

    def start(self):
        self.rodando = True
        t = threading.Thread(target=self.update)
        t.daemon = True
        t.start()
        return self

    def update(self):
        while self.rodando:
            ret, frame = self.stream.read()
            if ret:
                with self.lock:
                    self.ultimo_frame = frame
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.ultimo_frame is not None:
                return self.ultimo_frame.copy()
            return None


# FUNÇÕES DE DADOS (PICKLE + DB)
def carregar_dados():
    global lista_encodings, lista_nomes
    try:
        # 1. Lê os bytes criptografados do disco
        with open(ARQUIVO_DADOS, "rb") as f:
            dados_cifrados = f.read()
            
        # 2. Descriptografar de volta para bytes legíveis usando a chave
        dados_em_bytes = fernet_cipher.decrypt(dados_cifrados)
        
        # 3. Desserializar: Reconstrói o dicionário na memória RAM (usa-se loads e não load)
        data = pickle.loads(dados_em_bytes)
        
        lista_encodings = data["encodings"]
        lista_nomes = data["names"]
        print(f"[IA] Carregados {len(lista_nomes)} vetores faciais descriptografados com sucesso.")
        
    except FileNotFoundError:
        lista_encodings = []
        lista_nomes = []
        print("[IA] Banco biométrico não encontrado. Iniciando vazio.")
    except Exception as e:
        # Se a chave for diferente, o Fernet dispara o erro: cryptography.fernet.InvalidToken
        print(f"[ERRO CRÍTICO LGPD] Falha de Segurança ao abrir a biometria: {e}")
        lista_encodings = []
        lista_nomes = []


def salvar_dados():
    global lista_encodings, lista_nomes
    data = {"encodings": lista_encodings, "names": lista_nomes}
    
    # 1. Serializar: Transforma o dicionário em bytes puros na memória RAM
    dados_em_bytes = pickle.dumps(data)
    
    # 2. Criptografar: O Fernet embaralha os bytes usando AES-128 (CBC/HMAC)
    dados_cifrados = fernet_cipher.encrypt(dados_em_bytes)
    
    # 3. Guardar no disco: O ficheiro final fica completamente ilegível
    with open(ARQUIVO_DADOS, "wb") as f:
        f.write(dados_cifrados)


def treinar_novas_fotos(nome, lista_fotos):
    global lista_encodings, lista_nomes
    rostos_extraidos = 0

    # Não criamos pastas, operamos direto na memória RAM
    for img in lista_fotos:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes, num_jitters=5)
        
        for enc in encs:
            with lock:
                # BIO-HASHING: Multiplica o vetor original (128,) pela Matriz (128x128)
                # O resultado é um vetor totalmente distorcido, mas seguro.
                enc_cancelavel = np.dot(enc, MATRIZ_PROJECAO)
                
                lista_encodings.append(enc_cancelavel)
                lista_nomes.append(nome)
                rostos_extraidos += 1
    
    # Validação de Segurança e Feedback
    if rostos_extraidos > 0:
        # Só cadastra no banco e salva o pickle se a IA capturou o rosto
        cadastrar_usuario_db(nome)
        salvar_dados()
        print(f"[IA] Sucesso! {rostos_extraidos} vetor(es) biométrico(s) salvo(s) para '{nome}'.")
    else:
        print(f"[ERRO IA] Rosto não detectado nas fotos de '{nome}'. Nenhuma biometria salva. Tente novamente com melhor iluminação.")


# INTERFACE & CLIQUES
def desenhar_interface(frame):
    cv2.rectangle(frame, (0, ALTURA_TELA - 100), (LARGURA_TELA, ALTURA_TELA), COR_BARRA_FUNDO, -1)
    cv2.rectangle(frame, (50, ALTURA_TELA - 80), (300, ALTURA_TELA - 20), COR_BTN_FUNDO, -1)
    cv2.rectangle(frame, (50, ALTURA_TELA - 80), (300, ALTURA_TELA - 20), (255, 255, 255), 1)
    cv2.putText(frame, "Capturar", (110, ALTURA_TELA - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_TEXTO, 2)

    cv2.rectangle(frame, (350, ALTURA_TELA - 80), (600, ALTURA_TELA - 20), COR_BTN_FUNDO, -1)
    cv2.rectangle(frame, (350, ALTURA_TELA - 80), (600, ALTURA_TELA - 20), (255, 255, 255), 1)
    cv2.putText(frame, "Envio Remoto", (390, ALTURA_TELA - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_TEXTO, 2)


def gerenciar_cliques(event, x, y, flags, param):
    global estado_atual, nome_novo_cadastro, buffer_fotos_novas
    y_min = ALTURA_TELA - 80
    y_max = ALTURA_TELA - 20

    if event == cv2.EVENT_LBUTTONDOWN:
        if estado_atual == MODO_RECONHECIMENTO:
            if y_min < y < y_max:
                if 50 < x < 300:
                    estado_atual = MODO_CAPTURANDO
                    nome_novo_cadastro = ""
                    buffer_fotos_novas = []
                elif 350 < x < 600:
                    estado_atual = MODO_INFO_REMOTO
        elif y < (ALTURA_TELA - 100):
            estado_atual = MODO_RECONHECIMENTO


# FUNÇÃO QUE COMPUTA A IA EM BACKGROUND (THREAD PARALELA)
def processar_ia_async(frame_ia, frame_cru_ia):
    global ia_processando, caixas_detectadas, nomes_detectados, ultimo_sucesso, nome_detectado
    global ultimo_nome_reconhecido  # Permite ler e alterar a memória de estado do Totem
    
    try:
        inicio_inferencia = time.time()
        agora = inicio_inferencia

        small = cv2.resize(frame_ia, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs_small = face_recognition.face_locations(rgb_small)
        
        # GATILHO 1: Se a tela estiver completamente vazia, limpa o estado de memória
        if not locs_small:
            with lock:
                ultimo_nome_reconhecido = None
                caixas_detectadas = []
                nomes_detectados = []
            return

        locs_full = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in locs_small]
        rgb_full = cv2.cvtColor(frame_ia, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb_full, locs_full, num_jitters=1)

        novas_caixas = locs_small
        novos_nomes = []

        for enc in encs:
            name = "Desconhecido"
            with lock:
                enc_cancelavel = np.dot(enc, MATRIZ_PROJECAO)
                face_distances = face_recognition.face_distance(lista_encodings, enc_cancelavel)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    distancia_minima = face_distances[best_match_index]

                    if distancia_minima < 0.45:
                        name = lista_nomes[best_match_index]

            # GATILHO 2: Verificação e Controle de Duplicação de Log
            with lock:
                # Só gera log se o rosto atual for DIFERENTE do último que guardámos
                if name != ultimo_nome_reconhecido:
                    tempo_inferencia_ms = int((time.time() - inicio_inferencia) * 1000)
                    confianca_pct = round((1.0 - distancia_minima) * 100, 2) if len(face_distances) > 0 else 0.0
                    
                    # Regista o acesso de forma segura no SQLite
                    registrar_acesso_db(name, confianca_pct, frame_cru_ia.copy(), tempo_inferencia_ms)
                    
                    # Atualiza a memória para evitar novos logs repetidos deste utilizador
                    ultimo_nome_reconhecido = name  
                    ultimo_sucesso = agora
                    nome_detectado = name

            novos_nomes.append(name)

        with lock:
            caixas_detectadas = novas_caixas
            nomes_detectados = novos_nomes

    except Exception as e:
        print(f"[ERRO THREAD IA] {e}")
    finally:
        ia_processando = False

# LOOP PRINCIPAL (THREAD PRINCIPAL - FOCO EM FLUIDEZ VISUAL)
def loop_principal():
    global frame_atual, estado_atual, nome_novo_cadastro, buffer_fotos_novas
    global ia_processando, caixas_detectadas, nomes_detectados, ultimo_sucesso, nome_detectado

    stream = VideoStream(URL_CAMERA).start()
    time.sleep(2)

    cv2.namedWindow("Totem", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Totem", gerenciar_cliques)

    cv2.resizeWindow("Totem", LARGURA_TELA, ALTURA_TELA)
    cv2.moveWindow("Totem", 0, 0)
    cv2.setWindowProperty("Totem", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ultimo_ia = 0
    meu_ip = obter_ip_local()

    while True:
        try:
            frame_cru = stream.read()
            if frame_cru is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame_cru, (LARGURA_TELA, ALTURA_TELA))

            if estado_atual == MODO_RECONHECIMENTO:
                desenhar_interface(frame)

                agora = time.time()
                if pessoa_na_frente:
                    # Verifica o temporizador e se não há nenhuma análise ativa em background
                    if not ia_processando and (agora - ultimo_ia) > INTERVALO_SCAN_IA:
                        ultimo_ia = agora
                        ia_processando = True
                        
                        # Dispara a thread secundária para computar a IA de forma assíncrona
                        t_ia = threading.Thread(
                            target=processar_ia_async, 
                            args=(frame.copy(), frame_cru.copy())
                        )
                        t_ia.daemon = True
                        t_ia.start()

                    # Faz uma cópia rápida sob proteção do lock para renderizar as caixas na tela
                    with lock:
                        caixas_locais = caixas_detectadas.copy()
                        nomes_locais = nomes_detectados.copy()

                    for (top, right, bottom, left), name in zip(caixas_locais, nomes_locais):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cor = COR_RECONHECIDO if name != "Desconhecido" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), cor, cv2.FILLED)
                        cv2.putText(
                            frame,
                            name,
                            (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            1,
                        )
                else:
                    with lock:
                        caixas_detectadas = []
                        nomes_detectados = []
                    cv2.putText(
                        frame,
                        "Aproxime-se do Totem para liberar acesso",
                        (150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )

                if (agora - ultimo_sucesso) < DELAY_RECONHECIMENTO:
                    tempo_restante = int(DELAY_RECONHECIMENTO - (agora - ultimo_sucesso))
                    cv2.rectangle(frame, (0, 0), (LARGURA_TELA, 80), COR_RECONHECIDO, -1)
                    cv2.putText(
                        frame,
                        f"ACESSO LIBERADO: {nome_detectado}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 0),
                        3,
                    )
                    cv2.putText(
                        frame,
                        f"Aguarde {tempo_restante}s...",
                        (LARGURA_TELA - 250, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )

            elif estado_atual == MODO_CAPTURANDO:
                cv2.rectangle(frame, (0, 0), (LARGURA_TELA, 120), (200, 100, 0), -1)
                msg_nome = f"NOME: {nome_novo_cadastro}_"
                cv2.putText(frame, msg_nome, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COR_TEXTO, 2)
                info = f"[ESPACO] FOTO ({len(buffer_fotos_novas)})  |  [ENTER] SALVAR  |  [ESC] VOLTAR"
                cv2.putText(frame, info, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            elif estado_atual == MODO_INFO_REMOTO:
                cv2.rectangle(frame, (200, 200), (LARGURA_TELA - 200, ALTURA_TELA - 200), (0, 0, 0), -1)
                cv2.rectangle(frame, (200, 200), (LARGURA_TELA - 200, ALTURA_TELA - 200), (255, 255, 255), 2)
                cv2.putText(frame, "MODO SERVIDOR", (320, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COR_RECONHECIDO, 2)
                cv2.putText(frame, f"Servidor ativo em: {meu_ip}:5000", (240, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, "- Relatorio: /api/relatorio", (230, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_TEXTO, 2)
                cv2.putText(frame, "- Cadastro:  /api/cadastrar_direto", (230, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COR_TEXTO, 2)

            cv2.imshow("Totem", frame)
            with lock:
                frame_atual = frame.copy()

            key = cv2.waitKey(1) & 0xFF
            if estado_atual == MODO_CAPTURANDO:
                if key == 13:
                    if buffer_fotos_novas and nome_novo_cadastro:
                        treinar_novas_fotos(nome_novo_cadastro, buffer_fotos_novas)
                        estado_atual = MODO_RECONHECIMENTO
                elif key == 27:
                    estado_atual = MODO_RECONHECIMENTO
                elif key == 32:
                    buffer_fotos_novas.append(frame_cru.copy())
                elif key == 8:
                    nome_novo_cadastro = nome_novo_cadastro[:-1]
                elif 32 <= key <= 126:
                    nome_novo_cadastro += chr(key)

            if key == 9:
                break

        except Exception as e:
            time.sleep(0.01)

    stream.stop()
    cv2.destroyAllWindows()


# API FLASK
@app.route("/api/cadastrar_direto", methods=["POST"])
def cadastrar_direto():
    global lista_encodings, lista_nomes
    if "foto" not in request.files or "nome" not in request.form:
        return jsonify({"erro": "Dados incompletos"}), 400

    file = request.files["foto"]
    name = request.form["nome"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"erro": "Imagem invalida ou corrompida"}), 400

    altura, largura = img.shape[:2]
    if largura > 800:
        proporcao = 800.0 / largura
        img = cv2.resize(img, (800, int(altura * proporcao)))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with lock:
        boxes = face_recognition.face_locations(rgb)
        if boxes:
            encs = face_recognition.face_encodings(rgb, boxes, num_jitters=5)
            if len(encs) > 0:
                cadastrar_usuario_db(name)
                lista_encodings.append(encs[0])
                lista_nomes.append(name)
                salvar_dados()
                return jsonify({"msg": f"Sucesso! {name} cadastrado."}), 201

    return jsonify({"erro": "Rosto nao encontrado na foto"}), 400


@app.route("/api/relatorio", methods=["GET"])
def relatorio_acessos():
    conn = sqlite3.connect(BANCO_DADOS)
    c = conn.cursor()
    c.execute("""
        SELECT u.nome, l.data_hora, l.confianca_reconhecimento, l.foto_momento,
               l.status_acesso, l.tempo_inferencia_ms
        FROM Logs_Acesso l
        JOIN Usuarios u ON l.usuario_id = u.id
        ORDER BY l.data_hora DESC LIMIT 100
    """)
    logs = []
    for row in c.fetchall():
        logs.append(
            {
                "usuario": row[0],
                "data_hora": row[1],
                "confianca_pct": row[2],
                "status_acesso": row[4] if row[4] else "LIBERADO",
                "tempo_inferencia_ms": row[5] if row[5] else 0,
                "foto_caminho": row[3],
            }
        )
    conn.close()
    return jsonify(logs)


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with lock:
                if frame_atual is None:
                    time.sleep(0.1)
                    continue
                _, enc = cv2.imencode(".jpg", frame_atual)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(enc) + b"\r\n")

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def rodar_servidor():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    iniciar_banco()
    carregar_dados()

    t_flask = threading.Thread(target=rodar_servidor)
    t_flask.daemon = True
    t_flask.start()

    loop_principal()