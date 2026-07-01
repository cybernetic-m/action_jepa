import matplotlib.pyplot as plt
import numpy as np

# 1. Dati estratti dai risultati JSON
tasks = np.arange(10) # Task IDs da 0 a 9
task_labels = [f"{i}" for i in tasks]

sr_actor = [46.0, 100.0, 56.0, 28.0, 86.0, 80.0, 46.0, 98.0, 88.0, 32.0]
sr_refiner = [54.0, 98.0, 90.0, 70.0, 100.0, 94.0, 66.0, 100.0, 94.0, 56.0]
sr_ara = [70.0, 98.0, 84.0, 66.0, 88.0, 88.0, 70.0, 100.0, 100.0, 60.0]

# 2. Configurazione dello stile accademico
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

fig, ax = plt.subplots(figsize=(10, 5))

# Larghezza delle barre
bar_width = 0.25

# Posizioni sull'asse X per i tre gruppi
r1 = tasks - bar_width
r2 = tasks
r3 = tasks + bar_width

# 3. Creazione delle barre
ax.bar(r1, sr_actor, color='#4c72b0', width=bar_width, edgecolor='black', label='Actor (Baseline)')
ax.bar(r2, sr_refiner, color='#dd8452', width=bar_width, edgecolor='black', label='Refiner')
ax.bar(r3, sr_ara, color='#55a868', width=bar_width, edgecolor='black', label='Actor-Refiner Average (ARA)')

# 4. Personalizzazione degli assi
ax.set_xlabel('Task ID (Libero-Goal)', fontweight='bold', fontsize=14)
ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=14)
ax.set_xticks(tasks)
ax.set_xticklabels(task_labels)
ax.set_ylim([0, 110]) # Spazio per la legenda

# 5. Griglia e Legenda
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

# 6. Salvataggio in alta risoluzione per LaTeX (PNG)
plt.tight_layout()
plt.savefig('libero_goal_sr_barplot.png', dpi=300, bbox_inches='tight')

print("Grafico salvato con successo come 'libero_goal_sr_barplot.png'")