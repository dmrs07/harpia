# Harpia 🦅

> **"Fala português, entende o mundo"**

Harpia é uma LLM leve, brasileira e de código aberto — fine-tuned sobre o Llama 3.2 1B para falar PT-BR com personalidade, gírias e contexto cultural brasileiro.

Roda em qualquer máquina moderna (8GB RAM, CPU ou GPU).

---

## Personas

| Persona | Estilo |
|---|---|
| `zueiro` | Descontraído, bem-humorado, gírias brasileiras |
| `profissional` | Formal, objetivo, linguagem corporativa |
| `professor` | Didático, paciente, exemplos do cotidiano brasileiro |

---

## Instalação rápida

### Pré-requisito: [Ollama](https://ollama.com)

```bash
# Instala o Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh
```

### Rodar o Harpia

```bash
ollama run hf.co/dmrs07/harpia-gguf:Q4_K_M
```

---

## Usar via API (OpenAI-compatible)

O Ollama expõe automaticamente uma API compatível com OpenAI em `http://localhost:11434`.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="hf.co/dmrs07/harpia-gguf:Q4_K_M",
    messages=[
        {"role": "system", "content": "Você é o Harpia, uma IA brasileira bem-humorada."},
        {"role": "user", "content": "Explica o que é inflação de um jeito divertido."},
    ],
)

print(response.choices[0].message.content)
```

Instale o SDK: `pip install openai`

---

## Criar com persona via Modelfile

```bash
# Clona o repositório
git clone https://github.com/dmrs07/harpia
cd harpia

# Cria o modelo com persona zueiro
ollama create harpia -f Modelfile

# Conversa
ollama run harpia
```

---

## Stack técnica

| Componente | Tecnologia |
|---|---|
| Modelo base | Llama 3.2 1B Instruct |
| Fine-tuning | QLoRA via Unsloth |
| Infraestrutura de treino | Kaggle (2x T4, gratuito) |
| Runtime | Ollama |
| Formato | GGUF Q4_K_M |
| Armazenamento | Hugging Face |

---

## Estrutura do repositório

```
harpia/
├── Modelfile               # Configuração do modelo para o Ollama
├── training_data.jsonl     # Dataset de fine-tuning (formato ChatML)
├── harpia_full_pipeline.py # Pipeline completo: treino → export → HuggingFace
├── harpia_client.py        # Cliente Python com as 3 personas
├── generate_dataset.py     # Gerador de dataset sintético
└── README.md
```

---

## Roadmap

### v0.1 (atual)
- [x] Fine-tuning com 3 personas via system prompt
- [x] Publicado no HuggingFace (GGUF Q4_K_M)
- [x] API OpenAI-compatible via Ollama
- [x] Roda localmente com GPU de consumidor

### v0.2
- [ ] Dataset expandido (1500+ conversas)
- [ ] Avaliação automática por persona
- [ ] Skill packs: `redacao`, `programacao`, `cozinha`
- [ ] Connectors: WhatsApp, Pix

### v0.3
- [ ] LoRA adapter por persona (em vez de system prompt)
- [ ] Interface web local
- [ ] Documentação completa

---

## Contribuindo

Contribuições são bem-vindas! Abra uma issue ou PR.

Áreas prioritárias:
- Novos exemplos no dataset de treino
- Novos skill packs
- Connectors para serviços brasileiros
- Testes e avaliações

---

## Licença

Pesos: [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
Código: MIT
Modelo base: [Llama 3.2 Community License](https://ai.meta.com/llama/license/)

---

*Feito com 🇧🇷 por [dmrs07](https://github.com/dmrs07)*
