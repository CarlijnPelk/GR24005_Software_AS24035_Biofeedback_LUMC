#%% IMPORTEREN PACKAGES-----------------------------------------------------------------------------------------
import os
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# PAD NAAR HET VERIFICATIE BESTAND-----------------------------------------------------------------------------
target_file = os.path.join(os.getcwd(), 'verification_results3.json')

# DATA LADEN---------------------------------------------------------------------------------------------------
with open(target_file, 'r') as f:
    results = json.load(f)

# EXTRAHEER PARTICIPANTEN, SENSITIVITEITEN EN SPECIFICITEITEN------------------------------------------------
participants = [r['participant'] for r in results]
sensitivities = np.array([r['sensitivity'] for r in results], dtype=float)
specificities = np.array([r['specificity'] for r in results], dtype=float)

# DEFINIEER FUNCTIE OM GEMIDDELDE, SD EN 95%-CI TE BEREKENEN-------------------------------------------------
def summary_stats(data, alpha=0.05):
    n = len(data)
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    return mean, sd, (ci_lower, ci_upper)

# BEREKEN WAARDEN--------------------------------------------------------------------------------------------
sens_mean, sens_sd, sens_ci = summary_stats(sensitivities)
spec_mean, spec_sd, spec_ci = summary_stats(specificities)

sens_min, sens_max = float(np.min(sensitivities)), float(np.max(sensitivities))
spec_min, spec_max = float(np.min(specificities)), float(np.max(specificities))

# PRINTEN----------------------------------------------------------------------------------------------------
accuracy = np.mean(sensitivities + specificities) / 2
print(f"Accuracy: {accuracy:.3f}")

print("Participants included:", ", ".join(participants))
print(f"Sensitivity: mean={sens_mean:.3f}, SD={sens_sd:.3f}, 95% CI=[{sens_ci[0]:.3f}, {sens_ci[1]:.3f}], min={sens_min:.3f}, max={sens_max:.3f}")
print(f"Specificity: mean={spec_mean:.3f}, SD={spec_sd:.3f}, 95% CI=[{spec_ci[0]:.3f}, {spec_ci[1]:.3f}], min={spec_min:.3f}, max={spec_max:.3f}")

# WEERGAVE---------------------------------------------------------------------------------------------------
# Kies een witte rasterstijl
plt.style.use("seaborn-v0_8-whitegrid")

# Maak het figuur
fig, ax = plt.subplots(figsize=(8, 6))

# Kies kleurenpalet
boxprops      = dict(facecolor='#7C8DB7', edgecolor='#354C80', linewidth=1.5)
whiskerprops  = dict(color    ="#354C80", linewidth=1.5)
capprops      = dict(color    ='#354C80', linewidth=1.5)
medianprops   = dict(color    ="#354C80", linewidth=2)
flierprops    = dict(marker  ='o',
                      markerfacecolor='#354C80',
                      markeredgecolor  ='#354C80',
                      markersize=5,
                      alpha=0.7)
                   
# Teken boxplot
ax.boxplot(
    [sensitivities, specificities],
    labels=['Sensitivity', 'Specificity'],
    patch_artist=True,
    boxprops=boxprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    medianprops=medianprops,
    flierprops=flierprops
)

# Assen instellen
ax.set_ylabel('Proportion', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_ylim(0, 1) 

# Weergeven
plt.tight_layout()
plt.show()