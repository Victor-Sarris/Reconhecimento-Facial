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

# CONFIGURAÇÕES PRINCIPAIS
ARQUIVO_DADOS = "encodings.pickle"
BANCO_DADOS = "totem_banco.db"
PASTA_LOGS = "logs_imagens"
URL_CAMERA = 0
PASTA_DATASET = "dataset"
INTERVALO_SCAN_IA = 1.0
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

# MODULO A LASER (VL53L0X)

DISTANCIA_GATILHO_MM = 800  # 80 centímetros
pessoa_na_frente = True


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
    os.makedirs(PASTA_LOGS, exist_ok=True)
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
        c.execute(
            "ALTER TABLE Logs_Acesso ADD COLUMN status_acesso TEXT DEFAULT 'LIBERADO'"
        )
        c.execute(
            "ALTER TABLE Logs_Acesso ADD COLUMN tempo_inferencia_ms INTEGER DEFAULT 0"
        )
        c.execute("ALTER TABLE Logs_Acesso ADD COLUMN hardware_temp_c REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()
    print("[BANCO] Banco de Dados Inicializado com Sucesso.")


def cadastrar_usuario_db(nome, nivel="Aluno"):
    # Grava o usuário no banco de dados relacional.
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


# Funcao para registro de logs de acesso
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
    nome_arquivo = f"{PASTA_LOGS}/{agora_dt.strftime('%Y%m%d_%H%M%S')}_{nome.replace(' ', '_')}.jpg"
    cv2.imwrite(nome_arquivo, frame_capturado)

    c.execute(
        """
        INSERT INTO Logs_Acesso (usuario_id, data_hora, confianca_reconhecimento, foto_momento, status_acesso, tempo_inferencia_ms)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (user_id, agora_dt, confianca, nome_arquivo, "LIBERADO", tempo_ms),
    )

    conn.commit()
    print(
        f"[AUDITORIA] Acesso salvo: {nome} | Confiança: {confianca}% | Tempo: {tempo_ms}ms"
    )
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
        with open(ARQUIVO_DADOS, "rb") as f:
            data = pickle.load(f)
        lista_encodings = data["encodings"]
        lista_nomes = data["names"]
        print(f"[IA] Carregados {len(lista_nomes)} vetores faciais.")
    except FileNotFoundError:
        lista_encodings = []
        lista_nomes = []


def salvar_dados():
    global lista_encodings, lista_nomes
    data = {"encodings": lista_encodings, "names": lista_nomes}
    with open(ARQUIVO_DADOS, "wb") as f:
        f.write(pickle.dumps(data))


def treinar_novas_fotos(nome, lista_fotos):
    global lista_encodings, lista_nomes

    # Registra no Banco de Dados SQLite
    cadastrar_usuario_db(nome)

    pasta = os.path.join(PASTA_DATASET, nome)
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    count = len(os.listdir(pasta))
    for img in lista_fotos:
        filename = f"{pasta}/{count}.jpg"
        cv2.imwrite(filename, img)
        count += 1

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes, num_jitters=5)
        for enc in encs:
            with lock:
                lista_encodings.append(enc)
                lista_nomes.append(nome)

    salvar_dados()


# INTERFACE & CLIQUES
def desenhar_interface(frame):
    cv2.rectangle(
        frame, (0, ALTURA_TELA - 100), (LARGURA_TELA, ALTURA_TELA), COR_BARRA_FUNDO, -1
    )

    cv2.rectangle(
        frame, (50, ALTURA_TELA - 80), (300, ALTURA_TELA - 20), COR_BTN_FUNDO, -1
    )
    cv2.rectangle(
        frame, (50, ALTURA_TELA - 80), (300, ALTURA_TELA - 20), (255, 255, 255), 1
    )
    cv2.putText(
        frame,
        "Capturar",
        (110, ALTURA_TELA - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COR_TEXTO,
        2,
    )

    cv2.rectangle(
        frame, (350, ALTURA_TELA - 80), (600, ALTURA_TELA - 20), COR_BTN_FUNDO, -1
    )
    cv2.rectangle(
        frame, (350, ALTURA_TELA - 80), (600, ALTURA_TELA - 20), (255, 255, 255), 1
    )
    cv2.putText(
        frame,
        "Envio Remoto",
        (390, ALTURA_TELA - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COR_TEXTO,
        2,
    )


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


def alinhar_rosto(imagem_rbg):
    landmarks_lista = face_recognition.face_landmarks(imagem_rbg)

    # se nao achar nenhum rosto na imagem
    if not landmarks_lista:
        return imagem_rbg

    # se achar
    landmarks = landmarks_lista[0]

    # coordenadas dos olhos
    olho_esquerdo = landmarks["left_eye"]
    olho_direito = landmarks["right_eye"]

    # cacula o centro de cada olho
    centro_esq = np.mean(olho_esquerdo, axis=0).astype(int)
    centro_dir = np.mean(olho_direito, axis=0).astype(int)

    # calcula o angulo de inclinacao
    dY = centro_dir[1] - centro_esq[1]
    dX = centro_dir[0] - centro_esq[0]
    angulo = np.degrees(math.atan2(dY, dX))

    # calcula o ponto central
    eixo_rotacao = (
        int((centro_esq[0] + centro_dir[0]) / 2),
        int((centro_esq[1] + centro_dir[1]) / 2),
    )

    # rotaciona a imagem
    altura, largura = imagem_rbg.shape[:2]
    matriz_rotacao = cv2.getRotationMatrix2D(eixo_rotacao, angulo, 1.0)
    imagem_alinhada = cv2.warpAffine(
        imagem_rbg, matriz_rotacao, (largura, altura), flags=cv2.INTER_CUBIC
    )

    return imagem_alinhada


# LOOP PRINCIPAL
def loop_principal():
    global frame_atual, estado_atual, nome_novo_cadastro, buffer_fotos_novas

    stream = VideoStream(URL_CAMERA).start()
    time.sleep(2)

    cv2.namedWindow("Totem", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Totem", gerenciar_cliques)

    cv2.resizeWindow("Totem", LARGURA_TELA, ALTURA_TELA)
    cv2.moveWindow("Totem", 0, 0)
    cv2.setWindowProperty("Totem", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ultimo_ia = 0
    ultimo_sucesso = 0
    nome_detectado = ""
    caixas_detectadas = []
    nomes_detectados = []

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
                    if (agora - ultimo_ia) > INTERVALO_SCAN_IA:
                        ultimo_ia = agora

                        inicio_inferencia = time.time()

                        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                        locs_small = face_recognition.face_locations(rgb_small)
                        caixas_detectadas = locs_small
                        nomes_detectados = []

                        if locs_small:
                            locs_full = [
                                (top * 4, right * 4, bottom * 4, left * 4)
                                for (top, right, bottom, left) in locs_small
                            ]

                            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            encs = face_recognition.face_encodings(
                                rgb_full, locs_full, num_jitters=2
                            )

                            for enc in encs:
                                name = "Desconhecido"
                                with lock:
                                    face_distances = face_recognition.face_distance(
                                        lista_encodings, enc
                                    )
                                    if len(face_distances) > 0:
                                        best_match_index = np.argmin(face_distances)
                                        distancia_minima = face_distances[
                                            best_match_index
                                        ]

                                        if distancia_minima < 0.45:
                                            name = lista_nomes[best_match_index]

                                            em_cooldown = (
                                                agora - ultimo_sucesso
                                            ) < DELAY_RECONHECIMENTO
                                            if not em_cooldown:
                                                tempo_inferencia_ms = int(
                                                    (time.time() - inicio_inferencia)
                                                    * 1000
                                                )

                                                confianca_pct = round(
                                                    (1.0 - distancia_minima) * 100, 2
                                                )

                                                registrar_acesso_db(
                                                    name,
                                                    confianca_pct,
                                                    frame_cru.copy(),
                                                    tempo_inferencia_ms,
                                                )

                                                ultimo_sucesso = agora
                                                nome_detectado = name

                                nomes_detectados.append(name)
                        else:
                            nomes_detectados = []

                    for (top, right, bottom, left), name in zip(
                        caixas_detectadas, nomes_detectados
                    ):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cor = COR_RECONHECIDO if name != "Desconhecido" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)
                        cv2.rectangle(
                            frame, (left, bottom - 35), (right, bottom), cor, cv2.FILLED
                        )
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
                    tempo_restante = int(
                        DELAY_RECONHECIMENTO - (agora - ultimo_sucesso)
                    )
                    cv2.rectangle(
                        frame, (0, 0), (LARGURA_TELA, 80), COR_RECONHECIDO, -1
                    )
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
                cv2.putText(
                    frame,
                    msg_nome,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    COR_TEXTO,
                    2,
                )
                info = f"[ESPACO] FOTO ({len(buffer_fotos_novas)})  |  [ENTER] SALVAR  |  [ESC] VOLTAR"
                cv2.putText(
                    frame,
                    info,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

            elif estado_atual == MODO_INFO_REMOTO:
                cv2.rectangle(
                    frame,
                    (200, 200),
                    (LARGURA_TELA - 200, ALTURA_TELA - 200),
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    frame,
                    (200, 200),
                    (LARGURA_TELA - 200, ALTURA_TELA - 200),
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "MODO SERVIDOR",
                    (320, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    COR_RECONHECIDO,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Servidor ativo em: {meu_ip}:5000",
                    (240, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "- Relatorio: /api/relatorio",
                    (230, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COR_TEXTO,
                    2,
                )
                cv2.putText(
                    frame,
                    "- Cadastro:  /api/cadastrar_direto",
                    (230, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COR_TEXTO,
                    2,
                )

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

            if key == ord("q"):
                break

        except Exception as e:
            time.sleep(0.1)

    stream.stop()
    cv2.destroyAllWindows()


# API FLASK.
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


# a rota /api/relatorio exporta os logs do banco de dados em formato JSON
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
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + bytearray(enc)
                + b"\r\n"
            )

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def rodar_servidor():
    app.run(
        host="0.0.0.0", port=5000, debug=False, use_reloader=False
    )  # mudar a porta quase ja tenha um processo nessa


if __name__ == "__main__":
    iniciar_banco()
    carregar_dados()

    t_flask = threading.Thread(target=rodar_servidor)
    t_flask.daemon = True
    t_flask.start()

    loop_principal()
