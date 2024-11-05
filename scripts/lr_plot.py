import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def linear_warmup(x, start_lr, end_lr, warmup_tokens):
    return start_lr + (end_lr - start_lr) * (x / warmup_tokens)

def cosine_decay(x, start_lr, end_lr, total_tokens):
    progress = x / total_tokens
    return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * progress))

# Set up the token ranges
total_tokens = 432e9 # Pretending 4.32T is actually 432B
x = np.linspace(0, total_tokens+ 28.8e9, 10000)

# Main schedule
main_warmup = 5e9
main_decay_tokens = 432e9
main_warmup_lr = linear_warmup(np.minimum(x, main_warmup), 0, 0.01, main_warmup)
main_decay = cosine_decay(np.maximum(x - main_warmup, 0), 0.01, 0.00003, main_decay_tokens)
main_schedule = np.where(x <= main_warmup, main_warmup_lr, main_decay)
main_schedule = np.where(x <= total_tokens, main_schedule, np.nan)

# Other schedules
other_warmup = 1.4e9
other_total = 28.8e9
other_schedule1 = np.where(x <= other_warmup,
                           linear_warmup(x, 0, 0.003, other_warmup),
                           cosine_decay(x - other_warmup, 0.003, 0.00003, other_total - other_warmup))
other_schedule1 = np.where(x <= other_total, other_schedule1, np.nan)

other_schedule2 = np.where((x >= 432e9 - other_warmup) & (x <= 432e9 + other_warmup),
                           linear_warmup(x - 432e9, 0, 0.003, other_warmup),
                           cosine_decay(x - 432e9 - other_warmup, 0.003, 0.00003, other_total - other_warmup))
other_schedule2 = np.where((x >= 432e9) & (x <= 432e9 + other_total), other_schedule2, np.nan)

# Cosine decay curves
cosine_decays = []
for start_percentage in [0.2, 0.4, 0.6, 0.8]:
    start_x = 432e9 * start_percentage
    start_lr = cosine_decay(start_x - main_warmup, 0.01, 0.00003, main_decay_tokens)
    curve = cosine_decay(x - start_x, start_lr, 0.00003, 28.8e9)
    curve = np.where((x >= start_x) & (x <= start_x + 28.8e9), curve, np.nan)
    cosine_decays.append(curve)


# Color scheme
colors = {
    'main': '#4169E1',  # Royal Blue (keep)
    'other1': '#C44E52',  # Muted Red (adjusted)
    'other2': '#FF8C00',  # Softer Orange (adjusted)
    'cosine1': '#9467BD',  # Vibrant Purple (keep)
    'cosine2': '#E377C2',  # Cool Magenta (keep)
    'cosine3': '#20B2AA',  # Light Sea Green (keep)
    'cosine4': '#7CAB66'   # Jade Green (keep)
}
# colors = {
#     'main': '#4169E1',  # Royal Blue
#     'other1': '#DC143C',  # Crimson
#     'other2': '#FFA500',  # Orange
#     'cosine1': '#8A2BE2',  # Blue Violet
#     'cosine2': '#FF69B4',  # Hot Pink
#     'cosine3': '#20B2AA',  # Light Sea Green
#     'cosine4': '#7CAB66'  # Jade Green 
# }
# Set style
plt.style.use('seaborn-v0_8-pastel')

# Plotting
plt.figure(figsize=(8, 2))
plt.plot(x, main_schedule, label='Main Schedule', linewidth=2, color=colors['main'])
plt.plot(x, other_schedule1, label='Warmup-cosine 0%', linewidth=2, color=colors['other1'])
for i, curve in enumerate(cosine_decays):
    plt.plot(x, curve, label=f'Cosine {(i+1)*20}%', linewidth=2, color=colors[f'cosine{i+1}'])
plt.plot(x, other_schedule2, label='Warmup-cosine 100%', linewidth=2, color=colors['other2'])

# plt.yscale('log')
plt.xlabel('Token Count')
plt.ylabel('Learning Rate')
# plt.title('Learning Rate Schedules (4.32T main schedule)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Customize x-axis ticks
x_ticks = [0, 100e9, 200e9, 300e9, 400e9, total_tokens]
plt.xticks(x_ticks, ['0', '1T', '2T', '3T', '4T', f'{total_tokens/1e11:.1f}T'])
plt.xlim(-1000000000, total_tokens + 28.8e9)
plt.ylim(0, 0.011)

plt.tight_layout()
plt.savefig('learning_rate_schedules_adjusted.png', dpi=300)
plt.show()