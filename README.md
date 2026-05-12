# 📸 Totem de Reconhecimento Facial

> Sistema de controle de acesso inteligente.

## 📄 Sobre o Projeto

Este projeto visa modernizar o controle de entrada através de um totem de reconhecimento facial. O sistema utiliza **Redes Neurais Convolucionais (CNNs)** para analisar rostos em tempo real, gerando assinaturas vetoriais únicas para validar a identidade dos usuários de forma segura e eficiente.

Você pode encontrar a documentação dos scripts aqui: https://face-recognition-totem.netlify.app/#/

## 🛠️ Hardware

O projeto utiliza uma arquitetura híbrida para otimizar custos e desempenho:

- **Labrador 32-Bits:** Unidade central de processamento (SBC). Responsável por rodar os algoritmos de IA, gerenciar o banco de dados e a lógica de acesso.
- **ESP32-CAM:** Módulo de captura de imagem. Envia o fluxo de vídeo ou fotos estáticas para o processador central.
- **PAINEL IPS:** Painel IPS de 7 Polegadas para visualização do Reconhecimento.
- **Módulo VL530X:** Módulo de distância. Identifica quando uma pessoa está a frente da câmera.

## 🚀 Tecnologias

- **Linguagens:** Python (Backend/IA) e C++ (Firmware do ESP32).
- **Inteligência Artificial:** Extração de características faciais (_Face Embeddings_) via CNNs.
- **Comunicação:** Integração via rede (HTTP/WebSocket) entre a câmera e a placa Labrador.

## ⚙️ Como Funciona

1.  O **Módulo VL530X** identifica a pessoa próxima.
2.  O **ESP32-CAM** captura a imagem do visitante.
3.  A imagem é transmitida para a **Labrador**.
4.  O algoritmo processa a imagem, detecta a face e compara os vetores biométricos com o banco de dados.
5.  Se houver _match_ (correspondência positiva), o sistema aciona a liberação (ex: trava magnética ou catraca).

---

_Desenvolvido como Trabalho de Conclusão de Curso (TCC)._

<!-- Somos todos apenas pó de estrela (👉ﾟヮﾟ)👉 -->
