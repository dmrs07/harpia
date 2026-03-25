from openai import OpenAI

PERSONAS = {
    "zueiro": "Voce e Harpia, uma IA brasileira bem-humorada, direta e descontraida. Responde sempre em portugues brasileiro. Usa girias naturalmente, sem forcar. E inteligente mas nao se leva a serio.",
    "profissional": "Voce e Harpia, um assistente brasileiro no modo profissional. Comunica-se de forma clara, objetiva e formal. Foca em precisao e eficiencia.",
    "professor": "Voce e Harpia, um assistente brasileiro no modo Professor Coruja. E didatico, paciente e explicativo. Adora ensinar, usa exemplos do cotidiano brasileiro.",
}

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # valor ignorado pelo Ollama, mas obrigatório pelo SDK
)

def chat(mensagem, persona="zueiro"):
    response = client.chat.completions.create(
        model="hf.co/dmrs07/harpia-gguf:Q4_K_M",
        messages=[
            {"role": "system", "content": PERSONAS[persona]},
            {"role": "user", "content": mensagem},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=== Zueiro ===")
    print(chat("Qual a capital do Brasil?", persona="zueiro"))

    print("\n=== Profissional ===")
    print(chat("Qual a capital do Brasil?", persona="profissional"))

    print("\n=== Professor ===")
    print(chat("Qual a capital do Brasil?", persona="professor"))
