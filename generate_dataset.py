"""
Script para gerar dataset sintético do Harpia usando a API do Gemini (Google AI Studio).
Gera conversas nas 3 personas com tópicos variados.

Pré-requisitos:
    pip install google-generativeai

Uso:
    export GEMINI_API_KEY="sua-chave"
    python generate_dataset.py --total 1500 --output training_data.jsonl
"""
import json
import time
import argparse
import os
from google import genai
from google.genai import types

PERSONAS = {
    "zueiro": {
        "system": "Você é o Harpia, um assistente brasileiro no modo zueiro. Fala de forma descontraída, usa gírias brasileiras, faz piadas e não tem papas na língua. É bem-humorado mas não ofensivo.",
        "desc": "descontraído, gírias, humor, informal"
    },
    "profissional": {
        "system": "Você é o Harpia, um assistente brasileiro no modo profissional. Comunica-se de forma clara, objetiva e formal. Foca em precisão e eficiência. Adequado para contexto corporativo e técnico.",
        "desc": "formal, objetivo, técnico, corporativo"
    },
    "professor": {
        "system": "Você é o Harpia, um assistente brasileiro no modo Professor Coruja. É didático, paciente e explicativo. Adora ensinar, usa exemplos do cotidiano brasileiro, e adapta a explicação ao nível do aluno. Encoraja o aprendizado sem nunca fazer o aluno se sentir inferior.",
        "desc": "didático, exemplos, paciente, encoraja"
    },
}

TOPICS = [
    # Tecnologia
    "inteligência artificial", "machine learning", "Python", "JavaScript", "Docker",
    "Kubernetes", "cloud computing", "AWS", "Linux", "Git", "GitHub", "API REST",
    "banco de dados", "SQL", "NoSQL", "MongoDB", "PostgreSQL", "Redis",
    "segurança da informação", "criptografia", "redes de computadores", "TCP/IP",
    "blockchain", "Web3", "open source", "DevOps", "CI/CD", "microserviços",
    "programação orientada a objetos", "estruturas de dados", "algoritmos",
    "LLM", "GPT", "fine-tuning", "GGUF", "Ollama", "HuggingFace", "Kaggle",
    "data science", "análise de dados", "visualização de dados", "Pandas",
    "React", "Node.js", "Django", "FastAPI", "GraphQL", "WebSocket",
    "quantização de modelos", "LoRA", "Transformers", "embeddings", "RAG",
    "VS Code", "Jupyter Notebook", "terminal Linux", "bash scripting",
    # Finanças e economia
    "inflação", "taxa Selic", "PIB", "investimentos", "Tesouro Direto",
    "ações na bolsa", "fundos imobiliários", "criptomoedas", "Bitcoin",
    "imposto de renda", "FGTS", "INSS", "aposentadoria", "previdência privada",
    "cartão de crédito", "financiamento imobiliário", "microcrédito",
    "empreendedorismo", "MEI", "fluxo de caixa", "ROI", "startups",
    "Pix", "Open Finance", "bancos digitais", "educação financeira",
    "orçamento pessoal", "dívidas", "score de crédito", "consórcio",
    # Ciência
    "evolução darwiniana", "genética", "DNA", "células-tronco",
    "física quântica", "relatividade", "cosmologia", "buracos negros",
    "mudanças climáticas", "efeito estufa", "energias renováveis",
    "química orgânica", "tabela periódica", "reações químicas",
    "sistema solar", "astronomia", "física nuclear",
    "neurociência", "psicologia cognitiva", "comportamento humano",
    "vacinação", "sistema imunológico", "biotecnologia", "CRISPR",
    # Brasil e cultura
    "história do Brasil", "independência do Brasil", "era Vargas", "ditadura militar",
    "Constituição de 1988", "sistema eleitoral", "poderes da república",
    "Amazônia", "Cerrado", "Pantanal", "biodiversidade brasileira",
    "culinária brasileira", "feijoada", "acarajé", "churrasco gaúcho",
    "carnaval", "festa junina", "capoeira", "samba", "forró", "bossa nova",
    "futebol brasileiro", "Copa do Mundo 1970", "literatura brasileira",
    "Machado de Assis", "Guimarães Rosa", "Paulo Coelho",
    "saudade", "diversidade cultural brasileira",
    "São Paulo", "Rio de Janeiro", "Nordeste brasileiro", "Sul do Brasil",
    "povos indígenas do Brasil", "quilombolas", "imigração no Brasil",
    # Saúde e bem-estar
    "sistema imunológico", "SUS", "saúde mental",
    "ansiedade", "depressão", "meditação", "exercício físico",
    "nutrição", "dieta equilibrada", "diabetes", "hipertensão",
    "primeiros socorros", "sono saudável", "sedentarismo",
    "saúde bucal", "ergonomia no trabalho", "burnout",
    # Educação e carreira
    "como aprender rápido", "técnica Pomodoro", "leitura eficiente",
    "vestibular", "ENEM", "faculdade", "mercado de trabalho",
    "currículo", "entrevista de emprego", "soft skills", "liderança",
    "gestão do tempo", "produtividade", "trabalho remoto",
    "carreira em TI", "como programar do zero", "bootcamp de programação",
    "freelancing", "networking profissional", "LinkedIn",
    # Cotidiano e outros
    "como cozinhar arroz", "receita de feijão tropeiro", "dicas de organização",
    "como economizar energia elétrica", "reciclagem em casa", "sustentabilidade",
    "trânsito no Brasil", "transporte público", "aplicativos de mobilidade",
    "como planejar uma viagem", "tirar passaporte", "morar no exterior",
    "relacionamentos interpessoais", "comunicação não-violenta",
    "filosofia estoica", "filosofia budista", "psicologia positiva",
    "inteligência emocional", "resiliência", "autoconhecimento",
]

GENERATION_PROMPT = """Gere UMA conversa de treinamento para o assistente Harpia no modo {persona_name} ({persona_desc}).

Formato obrigatório — retorne APENAS o JSON puro, sem markdown, sem ```json, sem explicações:
{{"user": "<pergunta do usuário em português brasileiro>", "assistant": "<resposta do Harpia no estilo {persona_name}>"}}

Regras:
- A pergunta deve ser sobre: {topic}
- A resposta deve ter entre 80 e 250 palavras
- Deve soar natural e autêntico no estilo {persona_name}
- Use português brasileiro com expressões culturais quando cabível
- NÃO use markdown na resposta (sem **, sem #, sem listas com -)
- Retorne APENAS o JSON puro"""


def generate_conversation(client, model_id, persona_name, topic):
    persona = PERSONAS[persona_name]
    prompt = GENERATION_PROMPT.format(
        persona_name=persona_name,
        persona_desc=persona["desc"],
        topic=topic,
    )

    response = client.models.generate_content(model=model_id, contents=prompt)
    raw = response.text.strip()

    # Remove markdown code blocks se o modelo insistir
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)

    return {
        "messages": [
            {"role": "system", "content": persona["system"]},
            {"role": "user", "content": data["user"]},
            {"role": "assistant", "content": data["assistant"]},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1500)
    parser.add_argument("--output", type=str, default="training_data.jsonl")
    parser.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("ERRO: defina GEMINI_API_KEY ou use --api-key")
        print("Obtenha sua chave grátis em: https://aistudio.google.com/apikey")
        return

    client = genai.Client(api_key=args.api_key)
    model_id = "gemini-2.0-flash-lite"

    persona_names = list(PERSONAS.keys())
    per_persona = args.total // len(persona_names)

    # Carrega conversas existentes para não duplicar
    existing = set()
    try:
        with open(args.output) as f:
            for line in f:
                entry = json.loads(line)
                user_msg = next(m["content"] for m in entry["messages"] if m["role"] == "user")
                existing.add(user_msg[:60])
        print(f"Dataset existente: {len(existing)} conversas. Continuando...")
    except FileNotFoundError:
        print("Criando novo dataset...")

    generated = 0
    errors = 0

    with open(args.output, "a", encoding="utf-8") as f:
        for persona_name in persona_names:
            print(f"\nGerando {per_persona} conversas — persona: {persona_name}")
            count = 0
            topic_idx = 0

            while count < per_persona:
                topic = TOPICS[topic_idx % len(TOPICS)]
                topic_idx += 1

                try:
                    entry = generate_conversation(client, model_id, persona_name, topic)
                    user_msg = entry["messages"][1]["content"][:60]

                    if user_msg in existing:
                        continue

                    existing.add(user_msg)
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    count += 1
                    generated += 1

                    if generated % 50 == 0:
                        print(f"  {generated}/{args.total} conversas geradas...")

                    time.sleep(0.5)  # respeita rate limit do free tier

                except (json.JSONDecodeError, KeyError):
                    errors += 1
                    continue
                except Exception as e:
                    err_str = str(e)
                    # Respeita o retry delay sugerido pela API
                    if "retry_delay" in err_str and "seconds:" in err_str:
                        try:
                            delay = int(err_str.split("seconds:")[1].split("}")[0].strip()) + 2
                        except Exception:
                            delay = 60
                        print(f"  Rate limit — aguardando {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"  Erro: {err_str[:120]}")
                        time.sleep(5)
                    continue

    print(f"\nConcluído!")
    print(f"  Geradas: {generated}")
    print(f"  Erros ignorados: {errors}")
    print(f"  Dataset salvo em: {args.output}")


if __name__ == "__main__":
    main()
