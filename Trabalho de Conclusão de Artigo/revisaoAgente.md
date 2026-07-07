Olá, Victor. Como avaliador da sua banca e especialista em revisão acadêmica e científica, analisei o seu Artigo de Trabalho de Conclusão de Curso (TCC).

Gostaria de começar parabenizando-o pela excelente escolha do tema e pelo rigor metodológico. A aplicação da **Design Science Research (DSR)** para resolver um problema real do sistema de saúde, utilizando Computação de Borda (*Edge Computing*) com um hardware nacional (SBC Labrador), demonstra grande maturidade técnica e aderência ao perfil do curso de Tecnologia em Análise e Desenvolvimento de Sistemas. O trabalho tem um mérito científico inegável e grande potencial prático.

No entanto, por se tratar de um texto ainda em fase de rascunho/conclusão, há pontos importantes que precisam ser ajustados antes da submissão final. Dividi meu *feedback* técnico nas categorias abaixo para facilitar as suas correções.

---

### 1. Elementos Pré-textuais (Resumo e *Abstract*)

* **Problema principal:** O Resumo e o *Abstract* estão preenchidos com marcadores de texto (ex: `[DESCREVER O OBJETIVO PRINCIPAL DA PESQUISA]`).
* **Ação necessária:** Você precisa redigir o texto definitivo. Segundo a **ABNT NBR 6028:2021**, o resumo deve ser redigido em **parágrafo único**, sem recuo de primeira linha, contendo entre **150 e 250 palavras** (por se tratar de um artigo).
* **Sugestão de estrutura baseada no seu texto:** *"Este artigo tem como objetivo desenvolver e avaliar um artefato tecnológico de reconhecimento facial em arquitetura de borda (Edge Computing) para automatizar o credenciamento de pacientes em clínicas de saúde. A metodologia adotada foi a Design Science Research (DSR)..."* (Siga essa lógica extraindo os dados dos capítulos 4, 7 e 8).

### 2. Revisão Linguística e Adequação Acadêmica

O texto está muito bem articulado e a linguagem é formal, mas identifiquei a presença de **termos do Português de Portugal (PT-PT)**, o que foge ao padrão acadêmico brasileiro. Além disso, há pequenos erros de digitação.

* **Ajustes de vocabulário (Regionalismos/PT-PT):**
* Troque "equipa" por "equipe" (ocorre na Seção 8.1 e 9).
* Troque "partilha / partilhadas" por "compartilhamento / compartilhadas" (ocorre na Seção 3, H2 e H3, e Seção 8.3).
* Troque "recolhidos" por "coletados" (ocorre na Seção 7.3 e 8.1).
* Troque "manuseamento" por "manuseio" (ocorre na Seção 8.3).
* Troque "ecrã" por "tela" (se houver, certifique-se de manter "tela" ou "display").


* **Correções Ortográficas/Digitação:**
* **Seção 2 (Problema):** "...passível a falhas humanas..." $\rightarrow$ Sugestão: "...*suscetível a* falhas humanas..." ou "...*passível de* falhas humanas...".
* **Seção 3 (Hipóteses):** "...credenciamento dos *pacietes*..." $\rightarrow$ Corrigir para *pacientes*.
* **Seção 6.1:** "...sucetíveis a erros..." $\rightarrow$ Corrigir para *suscetíveis*.
* **Seção 6.1:** "...Conforme descrito por Braga et al. (2013). (2013), este processo..." $\rightarrow$ Remover a duplicidade do ano. Sugestão: *"Conforme descrito por Braga et al. (2013), este processo..."*



### 3. Coesão, Coerência e Estrutura Textual

* **Seção 1 (Introdução):** Assim como o resumo, a introdução ainda possui marcadores (ex: `[ÁREA DE CONHECIMENTO]`). Você precisa substituir esses marcadores pelo texto definitivo. Sugiro que você "puxe" partes da sua Seção 2 (Problema) e Seção 5 (Justificativa) para compor a introdução de forma fluida, apresentando o contexto pandêmico/pós-pandêmico e a evolução da biometria facial logo no início.
* **Seção 7 (Metodologia):** A explicação da DSR está excelente e muito bem fundamentada em Peffers et al. (2007). O encadeamento lógico entre as etapas da DSR e o que foi feito na prática está perfeito.

### 4. Conformidade com Normas Acadêmicas (ABNT)

* **Formatação das Figuras e Tabelas:** Pela ABNT, a fonte da figura deve vir logo abaixo dela. O seu texto faz isso muito bem (ex: `Fonte: Adaptado de (Taigman et al., 2014)`), mas lembre-se de padronizar o tamanho da fonte (geralmente tamanho 10) no documento final.
* **Citações Diretas:** Não localizei citações diretas longas (com mais de 3 linhas), mas caso decida inserir alguma na versão final do seu referencial teórico, lembre-se de aplicar o recuo de 4cm à esquerda, fonte tamanho 10, sem aspas, conforme a **ABNT NBR 10520:2023**.
* **Lista de Referências (Seção Final):** A sua lista está num formato "híbrido" que lembra um pouco a APA. Para adequar rigorosamente à **ABNT NBR 6023:2018**, faça os seguintes ajustes:
* Nomes dos autores devem vir em CAIXA ALTA.
* Os títulos das obras (livros ou nomes dos periódicos) devem estar em **negrito**.
* Para artigos de eventos (*Proceedings*), é necessário informar o local de publicação e a editora.
* *Exemplo de correção (TAIGMAN):* TAIGMAN, Y. et al. Deepface: Closing the gap to human-level performance in face verification. *In*: **Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**. Columbus: IEEE, 2014. p. 1701-1708.



### 5. Avaliação do Mérito Científico e Consistência Técnica

O mérito do seu trabalho é altíssimo. A construção de um artefato real usando um *Single Board Computer* para processamento na borda (*Edge Computing*) é um diferencial enorme. Tenho apenas duas sugestões de enriquecimento técnico para a sua defesa e escrita final:

1. **Fundamentação da LGPD e *Embeddings*:** Na seção 8.3 (ou nas considerações finais), você menciona rapidamente a privacidade e a LGPD. Como argumento de autoridade técnica perante a banca, explicite que o sistema armazena **apenas o vetor matemático (embedding)** e não a foto (imagem real) do paciente no banco de dados local. Isso garante *Privacy by Design*, pois é matematicamente inviável (ou extremamente custoso) fazer a engenharia reversa de um *embedding* de 128 bytes para recriar o rosto exato da pessoa. Isso eleva muito o peso científico do seu trabalho no quesito segurança da informação.
2. **Justificativa do SBC Labrador:** Você menciona o SBC Labrador como "arquitetura nacional", o que é excelente. Vale a pena dedicar um parágrafo na Seção 7.4.3 (Design e Desenvolvimento) para justificar brevemente *por que* o Labrador foi escolhido em detrimento de um *Raspberry Pi*, por exemplo (custo-benefício, fomento à tecnologia nacional, disponibilidade no mercado brasileiro, etc.).

### Resumo das Próximas Ações (Checklist para o Autor):

1. [ ] Remover os marcadores `[ ]` e redigir o Resumo, *Abstract* e Introdução finais.
2. [ ] Corrigir os termos em Português de Portugal para o padrão brasileiro.
3. [ ] Corrigir os pequenos erros de digitação (pacietes, sucetíveis, citação duplicada do ano).
4. [ ] Formatar a lista de Referências Bibliográficas para o padrão estrito da ABNT.
5. [ ] (Opcional, mas recomendado) Adicionar um parágrafo sobre a segurança do *embedding* em relação à LGPD.

Seu trabalho está no caminho certo para uma nota excelente. Faça esses ajustes de lapidação e o artigo estará pronto para submissão e defesa. Bom trabalho!