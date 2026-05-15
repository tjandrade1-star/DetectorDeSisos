# 🦷 3MolarAI v3.0

Sistema desktop de inteligência artificial para **detecção e rotulagem assistida de terceiros molares** em radiografias panorâmicas odontológicas.

Desenvolvido como projeto de especialização — Python + YOLOv8 + CustomTkinter.

---

## 📋 Funcionalidades

| Aba | Descrição |
|---|---|
| 🔍 **Detecção** | Carregue radiografias e analise automaticamente com o modelo YOLO treinado |
| ✍️ **Revisão Interativa** | Valide e corrija as predições da IA para gerar dataset de treino |
| ⚙️ **Treinamento** | Execute treino incremental ou do zero com monitoramento em tempo real |
| 🛠️ **Configurações** | Gerencie GPU, modelos e diretório de imagens |

## ⌨️ Atalhos (Aba de Revisão)

| Tecla | Ação |
| :--- | :--- |
| `S` | Salvar → Dataset de Treino |
| `N` | Negativo (confirmar ausência de molares) |
| `I` | Inconclusivo (revisão futura) |
| `P` | Pular imagem |
| `Z` | Desfazer última ação |

**Mouse:**
- Arrastar em área vazia → Criar caixa manual
- Arrastar caixa → Mover
- Puxador inferior direito → Redimensionar
- Clique direito em caixa → Excluir

---

## 🚀 Instalação e Uso

### Pré-requisitos
- Python 3.10+ com CUDA (opcional, para GPU)
- ~4GB de espaço em disco

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/3molarAI.git
cd 3molarAI

# Crie o ambiente virtual
python -m venv venv_gpu
venv_gpu\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### Executar

```bash
# Com ambiente virtual ativo:
python app.py

# Ou use o lançador (Windows):
Iniciar_3MolarAI.bat
```

### Distribuição Portátil (ALPHA)

Para gerar uma versão portátil com Python embutido (sem instalação):

```powershell
.\fabricar_alpha.ps1
```

Isso cria a pasta `3MolarAI_ALPHA/` com Python, dependências e o modelo prontos para rodar em qualquer Windows.

---

## 📁 Estrutura do Projeto

```
3molarAI/
├── app.py                  # Aplicação principal
├── requirements.txt        # Dependências Python
├── Iniciar_3MolarAI.bat    # Lançador Windows
├── fabricar_alpha.ps1      # Script de build portátil
├── yolo26n.pt              # Modelo base YOLO nano
├── dataset/
│   ├── dataset.yaml        # Configuração do dataset YOLO
│   ├── images/             # Imagens (ignorado pelo Git)
│   ├── labels/             # Anotações YOLO (.txt)
│   └── unlabeled/          # Fila de revisão (ignorado pelo Git)
└── model/
    └── versions/
        └── best.pt         # Modelo mais recente treinado
```

---

## 🛠️ Tecnologias

- **[Python](https://www.python.org/)** + **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)** — Interface desktop nativa
- **[YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)** — Detecção de objetos
- **[PyTorch](https://pytorch.org/)** — Backend de inferência e treino (CPU/CUDA)
- **[OpenCV](https://opencv.org/)** + **[Pillow](https://python-pillow.org/)** — Processamento de imagens

---

## 📊 Dataset

O dataset de treinamento é composto por radiografias panorâmicas com anotações em formato YOLO (`class x_center y_center width height`). Classe única: `third_molar`.

As imagens **não são distribuídas** neste repositório por questões de privacidade (dados clínicos).

---

## 📄 Licença

Projeto acadêmico — Especialização em Informática em Saúde.  
Desenvolvido por **Thiago José Domingues de Andrade**.
