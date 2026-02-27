import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

model_id = "openvla/openvla-7b"

try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Usiamo FP16, supportato dalla Titan Xp
        low_cpu_mem_usage=True, 
        device_map="auto",          # Gestisce lui l'allocazione
        trust_remote_code=True
    )
    print("Successo! Il modello è in memoria in modalità FP16.")
    
except Exception as e:
    print(f"Errore: {e}")
    # Se qui ricevi Out of Memory, significa che i 16GB sono troppi pochi