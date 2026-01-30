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
# ⚙️ CONFIGURAÇÕES (AJUSTADO PARA TELA 7")
# ==========================================================
ARQUIVO_DADOS = "encodings.pickle"
URL_CAMERA = "http://192.168.18.159/stream"
PASTA_DATASET = "dataset"
INTERVALO_IA = 1.0

# !!! CORREÇÃO AQUI !!!
# Resolução do LCD 7 Polegadas do Labrador
LARGURA_TELA = 1024
ALTURA_TELA = 600

# Cores e Layout
COR_BARRA_FUNDO = (180, 0, 0)  # Azul Escuro (BGR)
COR_BTN_FUNDO = (0, 0, 0)  # Preto
COR_TEXTO = (255, 255, 255)  # Branco
ALTURA_BARRA = 100  # Altura da barra inferior

# Estados
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
# 🎥 VÍDEO BUFFER
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
# 🧠 IA & DADOS
# ==========================================================
def carregar_dados():
    global lista_encodings, lista_nomes
    try:
        with open(ARQUIVO_DADOS, "rb") as f:
            data = pickle.load(f)
        lista_encodings = data["encodings"]
        lista_nomes = data["names"]
        print(f"[IA] Carregados {len(lista_nomes)} perfis.")
    except:
        lista_encodings = []
        lista_nomes = []


def salvar_dados():
    global lista_encodings, lista_nomes
    data = {"encodings": lista_encodings, "names": lista_nomes}
    with open(ARQUIVO_DADOS, "wb") as f:
        f.write(pickle.dumps(data))


def treinar_novas_fotos(nome, lista_fotos):
    global lista_encodings, lista_nomes
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


# ==========================================================
# 🎨 INTERFACE
# ==========================================================
def desenhar_interface(frame):
    # Desenha a barra azul no fundo (sempre visível)
    cv2.rectangle(
        frame,
        (0, ALTURA_TELA - ALTURA_BARRA),
        (LARGURA_TELA, ALTURA_TELA),
        COR_BARRA_FUNDO,
        -1,
    )

    # Coordenadas dos Botões
    y_btn = ALTURA_TELA - 80
    h_btn = 60
    w_btn = 250

    # Botão 1: Capturar (Esquerda)
    x1 = 50
    cv2.rectangle(
        frame, (x1, y_btn), (x1 + w_btn, y_btn + h_btn), COR_BTN_FUNDO, -1, cv2.LINE_AA
    )
    cv2.rectangle(
        frame, (x1, y_btn), (x1 + w_btn, y_btn + h_btn), (255, 255, 255), 2, cv2.LINE_AA
    )  # Borda branca
    cv2.putText(
        frame,
        "Capturar",
        (x1 + 60, y_btn + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COR_TEXTO,
        2,
    )

    # Botão 2: Remoto (Direita)
    x2 = 350
    cv2.rectangle(
        frame, (x2, y_btn), (x2 + w_btn, y_btn + h_btn), COR_BTN_FUNDO, -1, cv2.LINE_AA
    )
    cv2.rectangle(
        frame, (x2, y_btn), (x2 + w_btn, y_btn + h_btn), (255, 255, 255), 2, cv2.LINE_AA
    )  # Borda branca
    cv2.putText(
        frame,
        "Envio Remoto",
        (x2 + 30, y_btn + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COR_TEXTO,
        2,
    )


def gerenciar_cliques(event, x, y, flags, param):
    global estado_atual, nome_novo_cadastro, buffer_fotos_novas

    # Definição das áreas de clique (precisa bater com o desenho)
    y_min = ALTURA_TELA - 80
    y_max = ALTURA_TELA - 20

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[CLICK] X={x}, Y={y}")  # Debug para saber se o touch funciona

        # Se estiver no modo normal, verifica botões
        if estado_atual == MODO_RECONHECIMENTO:
            # Botão Capturar (50 a 300)
            if y_min < y < y_max and 50 < x < 300:
                print("-> Indo para Captura")
                estado_atual = MODO_CAPTURANDO
                nome_novo_cadastro = ""
                buffer_fotos_novas = []

            # Botão Remoto (350 a 600)
            elif y_min < y < y_max and 350 < x < 600:
                print("-> Indo para Remoto")
                estado_atual = MODO_INFO_REMOTO

        # Se estiver em outros modos, volta ao clicar fora
        elif y < (ALTURA_TELA - 100):
            estado_atual = MODO_RECONHECIMENTO


# ==========================================================
# 🔄 LOOP PRINCIPAL
# ==========================================================
def loop_principal():
    global frame_atual, estado_atual, nome_novo_cadastro, buffer_fotos_novas

    stream = VideoStream(URL_CAMERA).start()
    time.sleep(2)

    cv2.namedWindow("Totem", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Totem", gerenciar_cliques)

    # Força tamanho 1024x600 (Segredo para aparecer na tela de 7")
    cv2.resizeWindow("Totem", LARGURA_TELA, ALTURA_TELA)
    cv2.moveWindow("Totem", 0, 0)
    cv2.setWindowProperty("Totem", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ultimo_ia = 0
    ultimo_sucesso = 0
    nome_detectado = ""

    while True:
        try:
            frame_cru = stream.read()
            if frame_cru is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame_cru, (LARGURA_TELA, ALTURA_TELA))

            # 1. DESENHA A INTERFACE (BARRAS E BOTÕES)
            if estado_atual == MODO_RECONHECIMENTO:
                desenhar_interface(frame)  # <--- Agora desenha sempre!

                # Lógica de Reconhecimento
                agora = time.time()
                if (agora - ultimo_sucesso) < 5.0:
                    cv2.rectangle(frame, (0, 0), (LARGURA_TELA, 60), (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"BEM-VINDO: {nome_detectado}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                if (agora - ultimo_ia) > INTERVALO_IA:
                    ultimo_ia = agora
                    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb)
                    if locs:
                        encs = face_recognition.face_encodings(rgb, locs)
                        for enc in encs:
                            with lock:
                                matches = face_recognition.compare_faces(
                                    lista_encodings, enc, tolerance=0.5
                                )
                            if True in matches:
                                nome_detectado = lista_nomes[matches.index(True)]
                                ultimo_sucesso = agora

            # 2. MODO CAPTURA
            elif estado_atual == MODO_CAPTURANDO:
                cv2.rectangle(frame, (0, 0), (LARGURA_TELA, 150), (200, 0, 0), -1)
                texto_nome = (
                    f"NOME: {nome_novo_cadastro}_"
                    if not nome_novo_cadastro
                    else f"NOME: {nome_novo_cadastro}"
                )
                cv2.putText(
                    frame,
                    texto_nome,
                    (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "[ESPACE] FOTO  |  [ENTER] SALVAR  |  [ESC] SAIR",
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )
                cv2.putText(
                    frame,
                    f"FOTOS: {len(buffer_fotos_novas)}",
                    (LARGURA_TELA - 200, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            # 3. MODO REMOTO
            elif estado_atual == MODO_INFO_REMOTO:
                cv2.rectangle(
                    frame,
                    (100, 100),
                    (LARGURA_TELA - 100, ALTURA_TELA - 100),
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    frame,
                    (100, 100),
                    (LARGURA_TELA - 100, ALTURA_TELA - 100),
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "SERVIDOR REMOTO ATIVO",
                    (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    3,
                )
                cv2.putText(
                    frame,
                    f"IP: 192.168.18.149",
                    (150, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Toque na tela para voltar",
                    (150, 500),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (150, 150, 150),
                    2,
                )

            cv2.imshow("Totem", frame)
            with lock:
                frame_atual = frame.copy()

            key = cv2.waitKey(1) & 0xFF

            # Teclado para cadastro
            if estado_atual == MODO_CAPTURANDO:
                if key == 13:  # ENTER
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

        except:
            time.sleep(0.1)

    stream.stop()
    cv2.destroyAllWindows()


# ==========================================================
# 🌐 API
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
        return jsonify({"msg": "Cadastrado"}), 201
    return jsonify({"erro": "Rosto nao encontrado"}), 400


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
