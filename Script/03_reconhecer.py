import cv2
import face_recognition
import pickle
import numpy as np
import threading
import time
import requests
import os
from flask import Flask, Response, jsonify, request

# ==========================================================
# ⚙️ CONFIGURAÇÕES (MANTENDO AS ORIGINAIS)
# ==========================================================
ARQUIVO_DADOS = "encodings.pickle"
URL_CAMERA = "http://192.168.18.159/stream"  # IP DA SUA CÂMERA
PASTA_DATASET = "dataset"
INTERVALO_SCAN_IA = 1.0  # Otimização do 03_reconhecer.py
DELAY_RECONHECIMENTO = 5.0  # Tempo que a mensagem verde fica na tela

# Configurações de Tela (LCD 7")
LARGURA_TELA = 1024
ALTURA_TELA = 600

# Cores da Interface
COR_BARRA_FUNDO = (180, 0, 0)  # Azul Escuro
COR_BTN_FUNDO = (20, 20, 20)  # Preto suave
COR_TEXTO = (255, 255, 255)  # Branco
COR_RECONHECIDO = (0, 255, 0)  # Verde Matrix

# Estados do Sistema
MODO_RECONHECIMENTO = 0
MODO_CAPTURANDO = 1
MODO_INFO_REMOTO = 2

estado_atual = MODO_RECONHECIMENTO

# Globais
app = Flask(__name__)
lock = threading.Lock()
frame_atual = None
lista_encodings = []
lista_nomes = []
nome_novo_cadastro = ""
buffer_fotos_novas = []


# ==========================================================
# 🎥 VÍDEO STREAM (ORIGINAL DO 03_RECONHECER)
# ==========================================================
class VideoStream:
    def __init__(self, src):
        self.src = src
        self.stream = None
        self.bytes_buffer = bytes()
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
            try:
                if self.stream is None:
                    self.stream = requests.get(self.src, stream=True, timeout=5)

                for chunk in self.stream.iter_content(chunk_size=4096):
                    if not self.rodando:
                        if self.stream:
                            self.stream.close()
                        break
                    self.bytes_buffer += chunk
                    a = self.bytes_buffer.find(b"\xff\xd8")
                    b = self.bytes_buffer.find(b"\xff\xd9")
                    if a != -1 and b != -1:
                        jpg = self.bytes_buffer[a : b + 2]
                        self.bytes_buffer = self.bytes_buffer[b + 2 :]
                        img = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        with self.lock:
                            self.ultimo_frame = img
            except:
                self.stream = None
                self.bytes_buffer = bytes()
                time.sleep(2)

    def read(self):
        with self.lock:
            return self.ultimo_frame

    def stop(self):
        self.rodando = False


# ==========================================================
# 🧠 FUNÇÕES DE DADOS (ORIGINAL DO 02_TREINAR)
# ==========================================================
def carregar_dados():
    global lista_encodings, lista_nomes
    try:
        with open(ARQUIVO_DADOS, "rb") as f:
            data = pickle.load(f)
        lista_encodings = data["encodings"]
        lista_nomes = data["names"]
        print(f"[SISTEMA] Banco carregado: {len(lista_nomes)} usuarios.")
    except FileNotFoundError:
        print("[AVISO] Iniciando banco de dados vazio.")
        lista_encodings = []
        lista_nomes = []


def salvar_dados():
    global lista_encodings, lista_nomes
    data = {"encodings": lista_encodings, "names": lista_nomes}
    with open(ARQUIVO_DADOS, "wb") as f:
        f.write(pickle.dumps(data))


def treinar_novas_fotos(nome, lista_fotos):
    global lista_encodings, lista_nomes
    print(f"[TREINO] Processando {len(lista_fotos)} fotos para {nome}...")

    pasta = os.path.join(PASTA_DATASET, nome)
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    count = len(os.listdir(pasta))
    for img in lista_fotos:
        # Salva arquivo físico
        filename = f"{pasta}/{count}.jpg"
        cv2.imwrite(filename, img)
        count += 1

        # Gera encoding para memória (Treino Instantâneo)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)
        for enc in encs:
            with lock:
                lista_encodings.append(enc)
                lista_nomes.append(nome)

    salvar_dados()
    print("[TREINO] Concluído e Salvo!")


# ==========================================================
# 🎨 INTERFACE & CLIQUES
# ==========================================================
def desenhar_interface(frame):
    # Barra Inferior Azul
    cv2.rectangle(
        frame, (0, ALTURA_TELA - 100), (LARGURA_TELA, ALTURA_TELA), COR_BARRA_FUNDO, -1
    )

    # Botão Capturar
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

    # Botão Remoto
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
            # Clicar fora volta pro inicio
            estado_atual = MODO_RECONHECIMENTO


# ==========================================================
# 🔄 LOOP PRINCIPAL (INTEGRAÇÃO TOTAL)
# ==========================================================
def loop_principal():
    global frame_atual, estado_atual, nome_novo_cadastro, buffer_fotos_novas

    stream = VideoStream(URL_CAMERA).start()
    time.sleep(2)

    cv2.namedWindow("Totem", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Totem", gerenciar_cliques)

    # Configuração de Tela Cheia (Blindada)
    cv2.resizeWindow("Totem", LARGURA_TELA, ALTURA_TELA)
    cv2.moveWindow("Totem", 0, 0)
    cv2.setWindowProperty("Totem", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ultimo_ia = 0
    ultimo_sucesso = 0
    nome_detectado = ""
    # Cache de caixas para desenhar o quadrado suavemente entre frames
    caixas_detectadas = []
    nomes_detectados = []

    while True:
        try:
            frame_cru = stream.read()
            if frame_cru is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame_cru, (LARGURA_TELA, ALTURA_TELA))

            # --- LÓGICA DO 03_RECONHECER.PY INTEGRADA ---
            if estado_atual == MODO_RECONHECIMENTO:
                desenhar_interface(frame)

                agora = time.time()

                # Só processa IA se passou o intervalo (economiza CPU)
                if (agora - ultimo_ia) > INTERVALO_SCAN_IA:
                    ultimo_ia = agora

                    # Reduz imagem (Lógica original)
                    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                    # Detecta
                    locs = face_recognition.face_locations(rgb)
                    caixas_detectadas = locs  # Atualiza cache
                    nomes_detectados = []

                    if locs:
                        encs = face_recognition.face_encodings(rgb, locs)
                        for enc in encs:
                            name = "Desconhecido"
                            with lock:
                                # Lógica de Tolerância 0.5 (Original)
                                matches = face_recognition.compare_faces(
                                    lista_encodings, enc, tolerance=0.5
                                )
                                if True in matches:
                                    first_match_index = matches.index(True)
                                    name = lista_nomes[first_match_index]
                                    ultimo_sucesso = agora
                                    nome_detectado = name
                            nomes_detectados.append(name)
                    else:
                        nomes_detectados = []

                # --- DESENHO VISUAL (O INCREMENTO DA INTERFACE) ---

                # 1. Desenha quadrados nos rostos (Usando o cache)
                # Como reduzimos 0.25x (1/4), multiplicamos por 4 pra desenhar certo
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

                # 2. Barra Superior de Sucesso (Original do 03_reconhecer)
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

            # --- MODO CAPTURA (LÓGICA DO 01_CAPTURAR + INTERFACE) ---
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

            # --- MODO REMOTO ---
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
                    "Envie fotos para:",
                    (250, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    COR_TEXTO,
                    2,
                )
                # Tenta pegar IP real se possivel, senao mostra o fixo
                cv2.putText(
                    frame,
                    "http://192.168.18.149:5000",
                    (250, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Totem", frame)
            with lock:
                frame_atual = frame.copy()

            # --- CONTROLE DE TECLADO ---
            key = cv2.waitKey(1) & 0xFF

            if estado_atual == MODO_CAPTURANDO:
                if key == 13:  # ENTER
                    if buffer_fotos_novas and nome_novo_cadastro:
                        treinar_novas_fotos(nome_novo_cadastro, buffer_fotos_novas)
                        estado_atual = MODO_RECONHECIMENTO
                elif key == 27:
                    estado_atual = MODO_RECONHECIMENTO
                elif key == 32:  # ESPAÇO
                    buffer_fotos_novas.append(frame_cru.copy())
                elif key == 8:
                    nome_novo_cadastro = nome_novo_cadastro[:-1]
                elif 32 <= key <= 126:
                    nome_novo_cadastro += chr(key)

            if key == ord("q"):
                break

        except Exception as e:
            # print(e) # Debug
            time.sleep(0.1)

    stream.stop()
    cv2.destroyAllWindows()


# ==========================================================
# 🌐 API FLASK (ORIGINAL DO CADASTRAR_REMOTO)
# ==========================================================
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
            lista_encodings.append(encs[0])
            lista_nomes.append(name)
            salvar_dados()
        return jsonify({"msg": f"Sucesso! {name} cadastrado."}), 201
    return jsonify({"erro": "Rosto nao encontrado na foto"}), 400


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
    carregar_dados()
    t = threading.Thread(target=loop_principal)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
