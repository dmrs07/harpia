# ============================================================
# CÉLULA 1 — Instalação
# ============================================================
# !pip install "unsloth" --upgrade
# !pip install "torchao==0.9.0" --force-reinstall --no-deps
# !pip install llama-cpp-python
# (rode essa célula separada e reinicie o kernel antes de continuar)


# ============================================================
# CÉLULA 2 — Carregar modelo base
# ============================================================
from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


# ============================================================
# CÉLULA 3 — Aplicar LoRA
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


# ============================================================
# CÉLULA 4 — Carregar dataset
# ============================================================
import json
from datasets import Dataset

EOS_TOKEN = tokenizer.eos_token

harpia_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{}<|eot_id|>"

harpia_data = []
with open("/kaggle/input/harpia-training/training_data.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        msgs = entry["messages"]
        system = next(m["content"] for m in msgs if m["role"] == "system")
        user   = next(m["content"] for m in msgs if m["role"] == "user")
        asst   = next(m["content"] for m in msgs if m["role"] == "assistant")
        harpia_data.append({"system": system, "input": user, "output": asst})

dataset = Dataset.from_list(harpia_data)

def formatting_prompts_func(examples):
    texts = []
    for s, i, o in zip(examples["system"], examples["input"], examples["output"]):
        texts.append(harpia_prompt.format(s, i, o) + EOS_TOKEN)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"Dataset carregado: {len(dataset)} exemplos")


# ============================================================
# CÉLULA 5 — Treinar
# ============================================================
from unsloth import is_bfloat16_supported
from unsloth.trainer import UnslothTrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = UnslothTrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer_stats = trainer.train()
print("Treino concluído!")


# ============================================================
# CÉLULA 6 — Publicar GGUF no HuggingFace
# ============================================================
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("HF_TOKEN")

model.push_to_hub_gguf(
    "dmrs07/harpia-gguf",
    tokenizer,
    quantization_method = "q4_k_m",
    token = token,
)

print("Pronto! https://huggingface.co/dmrs07/harpia-gguf")
