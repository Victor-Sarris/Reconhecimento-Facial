import cv2
import face_recognition
import pickle
import numpy as np
import threading
import time
import requests
import os
import sqlite3
from datetime import datetime
from flask import Flask, Response, jsonify, request

# CONFIGURAÇÕES PRINCIPAIS
ARQUIVO_DADOS = "encodings.pickle"
BANCO_DADOS = "totem_banco.db"
PASTA_LOGS = "logs_imagens"
URL_CAMERA = "http://192.168.137.217/stream"
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


# BANCO DE DADOS E AUDITORIA
def iniciar_banco():
    os.makedirs(PASTA_LOGS, exist_ok=True)
    conn = sqlite3.connect(BANCO_DADOS)
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS Usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT UNIQUE,
            data_cadastro DATETIME,
            nivel_acesso TEXT
        )
    """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS Logs_Acesso (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER,
            data_hora DATETIME,
            confianca_reconhecimento REAL,
            foto_momento TEXT,
            FOREIGN KEY(usuario_id) REFERENCES Usuarios(id)
        )
    """
    )
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


def registrar_acesso_db(nome, confianca, frame_capturado):
    # Gera o log de acesso e salva a foto do momento exato
    conn = sqlite3.connect(BANCO_DADOS)
    c = conn.cursor()

    # Busca o ID do usuário
    c.execute("SELECT id FROM Usuarios WHERE nome = ?", (nome,))
    row = c.fetchone()

    if not row:
        c.execute(
            "INSERT INTO Usuarios (nome, data_cadastro, nivel_acesso) VALUES (?, ?, ?)",
            (nome, datetime.now(), "Migrado do Sistema Antigo"),
        )
        conn.commit()
        print(
            f"[BANCO] Sincronização automática: Usuário antigo '{nome}' adicionado ao novo banco."
        )
    else:
        user_id = row[0]

    agora_dt = datetime.now()

    nome_arquivo = f"{PASTA_LOGS}/{agora_dt.strftime('%Y%m%d_%H%M%S')}_{nome.replace(' ', '_')}.jpg"
    cv2.imwrite(nome_arquivo, frame_capturado)

    c.execute(
        """
        INSERT INTO Logs_Acesso (usuario_id, data_hora, confianca_reconhecimento, foto_momento)
        VALUES (?, ?, ?, ?)
    """,
        (user_id, agora_dt, confianca, nome_arquivo),
    )

    conn.commit()
    print(
        f"[AUDITORIA] Acesso salvo: {nome} | Confiança: {confianca}% | Foto: {nome_arquivo}"
    )

    conn.close()


# VÍDEO STREAM
# ========================================= CONFIGURACOES DO PC =========================================
class VideoStream:
    def __init__(self, src=0):
        self.src = src
        # Inicializa a captura de vídeo do OpenCV.
        # Se 'src' for 0, ele conecta na webcam padrão do notebook.
        self.stream = cv2.VideoCapture(self.src)
        self.ultimo_frame = None
        self.rodando = False
        self.lock = threading.Lock()  # evita conflito entre os threads

    def start(self):
        self.rodando = True
        # Inicia o Thread que vai ficar caputrando imagens em segundo plano
        t = threading.Thread(target=self.update)
        t.daemon = True
        t.start()
        return self

    def update(self):
        while self.rodando:
            # O .read() nativo do OpenCV retorna duas variaveis
            # 1. ret: booleano indicando se a leitura deu certo
            # 2. frame: a matriz da imagem capturada
            ret, frame = self.stream.read()
            if ret:
                # Se leu com sucesso, atualiza a variavel principal com seguranca
                with self.lock:
                    self.ultimo_frame = frame
            else:  # Se houver falha na leitura, aguarda uma fracao de segundo
                time.sleep(0.01)

    def read(self):
        # Retorna o frame mais recente quando o loop principal pedir
        with self.lock:
            return self.ultimo_frame

    def stop(self):
        self.rodando = False
        # Fecha a camera quando o loop terminar
        if self.stream is not None:
            self.stream.release()


# ========================================= FIM CONFIGURACOES DO PC =========================================
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
        encs = face_recognition.face_encodings(rgb, boxes)
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


# LOOP PRINCIPAL
def loop_principal():
    global frame_atual, estado_atual, nome_novo_cadastro, buffer_fotos_novas

    # stream = VideoStream(URL_CAMERA).start()
    stream = VideoStream(0).start()
    # stream = cv2.VideoCapture(0)
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

                if (agora - ultimo_ia) > INTERVALO_SCAN_IA:
                    ultimo_ia = agora

                    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    locs = face_recognition.face_locations(rgb)
                    caixas_detectadas = locs
                    nomes_detectados = []

                    if locs:
                        encs = face_recognition.face_encodings(rgb, locs)
                        for enc in encs:
                            name = "Desconhecido"
                            with lock:
                                face_distances = face_recognition.face_distance(
                                    lista_encodings, enc
                                )
                                if len(face_distances) > 0:
                                    best_match_index = np.argmin(face_distances)
                                    distancia_minima = face_distances[best_match_index]

                                    if distancia_minima < 0.5:
                                        name = lista_nomes[best_match_index]

                                        em_cooldown = (
                                            agora - ultimo_sucesso
                                        ) < DELAY_RECONHECIMENTO
                                        if not em_cooldown:
                                            confianca_pct = round(
                                                (1.0 - distancia_minima) * 100, 2
                                            )
                                            registrar_acesso_db(
                                                name, confianca_pct, frame_cru.copy()
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
                    (320, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    COR_RECONHECIDO,
                    2,
                )
                cv2.putText(
                    frame,
                    "API de Relatorios disponivel em:",
                    (250, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    COR_TEXTO,
                    2,
                )
                cv2.putText(
                    frame,
                    "http://192.168.18.149:5000/api/relatorio",
                    (230, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
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


# API FLASK
@app.route("/api/cadastrar_direto", methods=["POST"])
def cadastrar_direto():
    global lista_encodings, lista_nomes
    if "foto" not in request.files or "nome" not in request.form:
        return jsonify({"erro": "Dados incompletos"}), 400
    file = request.files["foto"]
    name = request.form["nome"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)

    if boxes:
        encs = face_recognition.face_encodings(rgb, boxes)
        with lock:
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
    c.execute(
        """
        SELECT u.nome, l.data_hora, l.confianca_reconhecimento, l.foto_momento
        FROM Logs_Acesso l
        JOIN Usuarios u ON l.usuario_id = u.id
        ORDER BY l.data_hora DESC LIMIT 100
    """
    )
    logs = []
    for row in c.fetchall():
        logs.append(
            {
                "usuario": row[0],
                "data_hora": row[1],
                "confianca_pct": row[2],
                "foto_caminho": row[3],
            }
        )
    conn.close()
    return jsonify({"total_logs": len(logs), "acessos": logs})


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


if __name__ == "__main__":
    iniciar_banco()
    carregar_dados()
    t = threading.Thread(target=loop_principal)
    t.daemon = True
    t.start()
    app.run(
        host="0.0.0.0", port=5000, debug=False
    )  # mudar a porta se já tiver uma aplicacao rodando (que é o meu caso)
