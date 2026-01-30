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
# ⚙️ CONFIGURAÇÕES GERAIS
# ==========================================================
ARQUIVO_DADOS = "encodings.pickle"
URL_CAMERA = "http://192.168.18.159/stream"  # IP DA SUA CÂMERA
PASTA_DATASET = "dataset"
INTERVALO_IA = 1.0  # Segundos entre reconhecimentos (poupa CPU)

# Resolução da Tela (Fixa para o seu monitor)
LARGURA_TELA = 1920
ALTURA_TELA = 1080

# Cores (BGR)
COR_FUNDO_BTN = (20, 20, 20)  # Cinza Escuro
COR_BTN_CAP = (0, 0, 0)  # Preto (Capturar)
COR_BTN_REM = (0, 0, 0)  # Preto (Remoto)
COR_TEXTO = (255, 255, 255)  # Branco
COR_DESTAQUE = (255, 0, 0)  # Azul

# Estados do Sistema
MODO_RECONHECIMENTO = 0
MODO_MENU = 1
MODO_CAPTURANDO = 2
MODO_INFO_REMOTO = 3

estado_atual = MODO_RECONHECIMENTO

# Variáveis Globais
app = Flask(__name__)
lock = threading.Lock()
frame_atual = None
lista_encodings = []
lista_nomes = []
nome_novo_cadastro = ""  # Para digitar o nome
buffer_fotos_novas = []  # Fotos temporárias antes de salvar


# ==========================================================
# 🎥 CLASSE DE VÍDEO (BUFFERIZADO)
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
# 🧠 FUNÇÕES DE IA (TREINAMENTO E CARREGAMENTO)
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
    print(f"[TREINO] Iniciando treinamento para: {nome}")

    # Cria pasta física
    pasta = os.path.join(PASTA_DATASET, nome)
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    count = len(os.listdir(pasta))

    for img in lista_fotos:
        # Salva disco
        filename = f"{pasta}/{count}.jpg"
        cv2.imwrite(filename, img)
        count += 1

        # Gera Encoding (Treino)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)

        for enc in encs:
            with lock:
                lista_encodings.append(enc)
                lista_nomes.append(nome)

    salvar_dados()
    print("[TREINO] Concluído!")


# ==========================================================
# 🎨 INTERFACE GRÁFICA (GUI)
# ==========================================================
def desenhar_botao(img, texto, x, y, w, h, cor_fundo, ativo=False):
    # Efeito de transparência e bordas arredondadas (simulado)
    overlay = img.copy()
    cor_final = cor_fundo
    if ativo:
        cor_final = (50, 50, 50)  # Mais claro se hover

    cv2.rectangle(overlay, (x, y), (x + w, y + h), cor_final, -1)

    # Borda azul (estilo da sua foto)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (200, 0, 0), 2)

    # Mistura para ficar semi-transparente
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    # Texto Centralizado
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 1.0
    espessura = 2
    (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, escala, espessura)
    tx = x + (w - largura_texto) // 2
    ty = y + (h + altura_texto) // 2
    cv2.putText(img, texto, (tx, ty), fonte, escala, COR_TEXTO, espessura)


def gerenciar_cliques(event, x, y, flags, param):
    global estado_atual, nome_novo_cadastro, buffer_fotos_novas

    # Largura e altura dos botões
    btn_w, btn_h = 300, 100
    btn_y = ALTURA_TELA - 150
    btn1_x = 50
    btn2_x = 400

    if event == cv2.EVENT_LBUTTONDOWN:
        if estado_atual == MODO_RECONHECIMENTO:
            # Qualquer clique abre o menu
            estado_atual = MODO_MENU

        elif estado_atual == MODO_MENU:
            # Checa Botão Capturar
            if btn1_x < x < btn1_x + btn_w and btn_y < y < btn_y + btn_h:
                estado_atual = MODO_CAPTURANDO
                nome_novo_cadastro = ""
                buffer_fotos_novas = []

            # Checa Botão Remoto
            elif btn2_x < x < btn2_x + btn_w and btn_y < y < btn_y + btn_h:
                estado_atual = MODO_INFO_REMOTO

            # Clique fora fecha menu
            elif y < btn_y:
                estado_atual = MODO_RECONHECIMENTO

        elif estado_atual == MODO_CAPTURANDO:
            # Clique para voltar se não estiver digitando
            pass

        elif estado_atual == MODO_INFO_REMOTO:
            estado_atual = MODO_MENU


# ==========================================================
# 🔄 LOOP PRINCIPAL
# ==========================================================
def loop_principal():
    global frame_atual, estado_atual, nome_novo_cadastro, buffer_fotos_novas

    stream = VideoStream(URL_CAMERA).start()
    time.sleep(2)  # Buffer encher

    cv2.namedWindow("Totem", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Totem", gerenciar_cliques)

    # Tenta fullscreen
    cv2.resizeWindow("Totem", LARGURA_TELA, ALTURA_TELA)
    cv2.setWindowProperty("Totem", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ultimo_ia = 0
    ultimo_sucesso = 0
    nome_detectado = ""

    while True:
        try:
            # 1. Pega Frame
            frame_cru = stream.read()
            if frame_cru is None:
                time.sleep(0.01)
                continue

            # Resize forçado para ocupar tela
            frame = cv2.resize(frame_cru, (LARGURA_TELA, ALTURA_TELA))

            # --- LÓGICA POR ESTADO ---

            # >>> MODO 0: RECONHECIMENTO (Padrão)
            if estado_atual == MODO_RECONHECIMENTO:
                agora = time.time()

                # Barra verde se reconhecido
                if (agora - ultimo_sucesso) < 5.0:
                    cv2.rectangle(frame, (0, 0), (LARGURA_TELA, 80), (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"BEM-VINDO: {nome_detectado}",
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        3,
                    )

                # Roda IA a cada X segundos
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
                                print(f"Reconhecido: {nome_detectado}")

            # >>> MODO 1: MENU (Botões)
            elif estado_atual == MODO_MENU:
                # Fundo escurecido
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, (0, 0), (LARGURA_TELA, ALTURA_TELA), (0, 0, 0), -1
                )
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Desenha Botões
                btn_y = ALTURA_TELA - 150
                desenhar_botao(frame, "Capturar", 50, btn_y, 300, 100, COR_BTN_CAP)
                desenhar_botao(frame, "Envio Remoto", 400, btn_y, 300, 100, COR_BTN_REM)

                cv2.putText(
                    frame,
                    "PAINEL DE CONTROLE",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    5,
                )
                cv2.putText(
                    frame,
                    "Toque na tela para voltar",
                    (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 200),
                    2,
                )

            # >>> MODO 2: CAPTURANDO (Digitar nome + Tirar fotos)
            elif estado_atual == MODO_CAPTURANDO:
                cv2.rectangle(frame, (0, 0), (LARGURA_TELA, 150), (200, 0, 0), -1)

                if not nome_novo_cadastro:
                    cv2.putText(
                        frame,
                        "DIGITE O NOME NO TECLADO",
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        3,
                    )
                    cv2.putText(
                        frame,
                        "Pressione ENTER para confirmar",
                        (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (200, 200, 200),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"CADASTRANDO: {nome_novo_cadastro}",
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        3,
                    )
                    cv2.putText(
                        frame,
                        f"Fotos tiradas: {len(buffer_fotos_novas)} (Pressione ESPACO para foto, ENTER para salvar)",
                        (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (200, 200, 200),
                        2,
                    )

            # >>> MODO 3: INFO REMOTO
            elif estado_atual == MODO_INFO_REMOTO:
                cv2.rectangle(
                    frame,
                    (200, 200),
                    (LARGURA_TELA - 200, ALTURA_TELA - 200),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    "MODO SERVIDOR ATIVO",
                    (300, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    COR_DESTAQUE,
                    4,
                )
                cv2.putText(
                    frame,
                    "Acesse pelo seu PC:",
                    (300, 500),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "http://192.168.18.149:5000/api/cadastrar_direto",
                    (300, 600),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Clique para voltar",
                    (300, 800),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (100, 100, 100),
                    2,
                )

            # Exibe
            cv2.imshow("Totem", frame)
            with lock:
                frame_atual = frame.copy()

            # CONTROLE DE TECLADO (Essencial para o cadastro manual)
            key = cv2.waitKey(1) & 0xFF

            if estado_atual == MODO_CAPTURANDO:
                # Se ainda não tem nome, captura teclado
                if key == 13:  # ENTER
                    if not nome_novo_cadastro:  # Se apertou enter sem nome, ignora
                        pass
                    elif len(buffer_fotos_novas) > 0:  # Se já tem fotos e nome, salva
                        treinar_novas_fotos(nome_novo_cadastro, buffer_fotos_novas)
                        estado_atual = MODO_RECONHECIMENTO
                    else:
                        # Confirmou o nome, agora vai tirar fotos
                        pass
                elif key == 27:  # ESC (Cancela)
                    estado_atual = MODO_MENU
                elif key == 32:  # ESPAÇO (Tira Foto)
                    if nome_novo_cadastro:
                        buffer_fotos_novas.append(frame_cru.copy())
                        print("Foto capturada!")
                elif key == 8:  # Backspace
                    nome_novo_cadastro = nome_novo_cadastro[:-1]
                elif 32 <= key <= 126:  # Letras e Números
                    nome_novo_cadastro += chr(key)

            elif key == ord("q"):
                break

        except Exception as e:
            time.sleep(0.1)
            # print(e)

    stream.stop()
    cv2.destroyAllWindows()


# ==========================================================
# 🌐 API FLASK (RODA EM PARALELO)
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


# ==========================================================
# 🚀 START
# ==========================================================
if __name__ == "__main__":
    carregar_dados()
    t = threading.Thread(target=loop_principal)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
