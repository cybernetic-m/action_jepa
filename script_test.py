import torch
from transformers import AutoConfig, AutoModelForVision2Seq

model_id = "openvla/openvla-7b"

# Caricamento in 4-bit usando bitsandbytes
try:
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    print("Successo! Il modello è in memoria (4-bit).")
except Exception as e:
    print(f"Errore OOM o hardware: {e}")