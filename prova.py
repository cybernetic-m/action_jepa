import torch

# 1. Carica il file .pth
percorso_file = 'results/results_alcor_5/2026_06_24__17_38/best_model.pth'
checkpoint = torch.load(percorso_file, map_location='cpu')

# 2. I file .pth possono contenere direttamente i pesi (state_dict) 
# oppure un dizionario più grande con epoca, loss, ecc.
# Estraiamo i pesi nel modo corretto:
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  # Il file è direttamente lo state_dict

# 3. Stampa tutte le chiavi (i nomi dei layer)
print(f"Trovate {len(state_dict.keys())} chiavi. Eccone l'elenco:\n")
for key in state_dict.keys():
    print(key)