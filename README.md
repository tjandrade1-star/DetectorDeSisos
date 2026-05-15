# DetectorDeSisos

Sistema desktop para detecção automatizada de terceiros molares em radiografias panorâmicas odontológicas, desenvolvido como projeto de especialização em Informática em Saúde.

A aplicação combina um modelo YOLOv8 treinado localmente com uma interface gráfica interativa que permite ao profissional revisar e corrigir as predições, gerando novos dados de treino de forma incremental.

---

## Funcionalidades

- **Detecção** — análise automática de radiografias com identificação dos dentes 18, 28, 38 e 48
- **Revisão interativa** — interface com canvas para validar, corrigir e anotar predições manualmente
- **Treinamento local** — treino incremental ou do zero com monitoramento em tempo real
- **Gestão de modelos** — versionamento local de pesos treinados com seleção automática do mais recente

---

## Requisitos

- Python 3.10 ou superior
- CUDA (opcional, para aceleração por GPU)
- ~4 GB de espaço em disco

---

## Instalação

```bash
git clone https://github.com/tjandrade1-star/DetectorDeSisos.git
cd DetectorDeSisos

python -m venv venv_gpu
venv_gpu\Scripts\activate

pip install -r requirements.txt
```

---

## Uso

```bash
python app.py
```

### Atalhos — aba de Revisão

| Tecla | Ação |
|---|---|
| `S` | Salvar anotação no dataset de treino |
| `N` | Marcar como negativo (sem molares) |
| `I` | Marcar como inconclusivo |
| `P` | Pular imagem |
| `Z` | Desfazer última ação |

**Interação com o canvas:**
- Arrastar em área vazia — cria nova caixa de anotação
- Arrastar caixa existente — move
- Puxador no canto inferior direito — redimensiona
- Clique direito sobre caixa — remove

---

## Estrutura do projeto

```
DetectorDeSisos/
├── app.py                   # Aplicação principal
├── requirements.txt         # Dependências
├── yolo26n.pt               # Pesos base YOLOv8 nano
├── dataset/
│   ├── dataset.yaml         # Configuração do dataset (formato YOLO)
│   ├── images/              # Imagens de treino e validação (não versionadas)
│   ├── labels/              # Anotações YOLO (.txt, uma por imagem)
│   └── unlabeled/           # Fila de imagens pendentes de revisão
└── model/
    └── versions/
        └── best.pt          # Modelo mais recente treinado
```

---

## Dataset

As anotações seguem o formato YOLO: `class x_center y_center width height` (valores normalizados). Classe única: `third_molar`.

As imagens radiográficas não são distribuídas neste repositório por questões de privacidade (dados clínicos de pacientes).

---

## Tecnologias

- [Python](https://www.python.org/) / [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — interface gráfica desktop
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detecção de objetos
- [PyTorch](https://pytorch.org/) — inferência e treinamento (CPU/CUDA)
- [OpenCV](https://opencv.org/) / [Pillow](https://python-pillow.org/) — processamento de imagens

---

Desenvolvido por Thiago José Domingues de Andrade — Especialização em Informática em Saúde
