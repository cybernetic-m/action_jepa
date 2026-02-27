from transformers import AutoModelForVision2Seq, AutoProcessor
import torch


# Caricamento in 4-bit usando bitsandbytes
try:
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    #attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    device_map="auto",
    load_in_4bit=True,
    ).to("cuda:0")
    print("Successo! Il modello è in memoria (4-bit).")
except Exception as e:
    print(f"Errore OOM o hardware: {e}")