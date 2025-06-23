#%%
# IMPORTEREN PACKAGES -----------------------------------------------------------------------------------------
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import json
import seaborn as sns
from scipy.ndimage import label
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import wilcoxon

# PARAMETERS ---------------------------------------------------------------------------------------------------
interval = 0.001
n_channels = 3
window = 500
threshold_samples = int(0.55 * window)

# DIALOOG POP-UP -----------------------------------------------------------------------------------------------
root = tk.Tk()
root.withdraw()

# Participant nummer kiezen
ptnr = simpledialog.askstring("Participant Info", "Participant nummer:")
if not ptnr:
    messagebox.showerror("Fout", "Geen participant nummer ingevoerd.")
    exit()

save = simpledialog.askstring("Save", "Wil je de sensitiviteit en specificiteit opslaan? (ja/nee):")

mapnaam = f"EMG_Output_ptnr{ptnr}"
huidige_map = os.getcwd()
pad_naar_map = os.path.join(huidige_map, mapnaam)

if not os.path.isdir(pad_naar_map):
    messagebox.showerror("Fout", f"❌ De map '{mapnaam}' bestaat niet.")
    exit()

# Meerdere bestanden selecteren 
selected_files = filedialog.askopenfilenames(
    initialdir=pad_naar_map, 
    title="Kies één of meerdere CSV-bestanden", 
    filetypes=[("CSV bestanden", "*.csv")]
)
selected_files = list(selected_files)  

if not selected_files:
    messagebox.showerror("Fout", "⚠️ Geen bestand(en) geselecteerd.")
    exit()

# Vooraf vragen of het bestand wel of geen cocontractie bevat
labels_per_file = {}
for file in selected_files:
    bestand = os.path.basename(file)
    antwoord = messagebox.askquestion("Co-contraction?", f"Bevat dit bestand co-contraction?\n\n{bestand}", icon='question')
    labels_per_file[file] = 1 if antwoord == 'yes' else 0  # 1 = wel co-contractie, 0 = geen


# MVIC INLADEN -----------------------------------------------------------------------------------------------
mvic_file = f"mvic_{ptnr}.json"
mvic_dir = os.path.join(os.getcwd(), "mvic")  
load_path = os.path.join(mvic_dir, mvic_file)

if os.path.exists(load_path):
    with open(load_path, 'r') as f:
        mvic_data = json.load(f)
else:
    print(f"❌ No MVIC file found for participant {ptnr}")
    mvic_data = {}

mvic_DM = mvic_data.get("abduction", {}).get("mvic_DM", 1)
mvic_LD = mvic_data.get("adduction", {}).get("mvic_LD", 1)
mvic_TM = mvic_data.get("adduction", {}).get("mvic_TM", 1)

# CCI & DM-THRESHOLDS INLADEN ---------------------------------------------------------------------------------
folder = "cci measurements"
fname  = f"cci_{ptnr}.json"
path   = os.path.join(folder, fname)

with open(path, 'r') as f:
    cci_values = json.load(f)

CCI_TM_0 = cci_values["0 graden extensie without co-contraction"]["CCI_TM"]
CCI_TM_0_CO = cci_values["0 graden extensie with co-contraction"]["CCI_TM_CO"]
CCI_LD_0 = cci_values["0 graden extensie without co-contraction"]["CCI_LD"]
CCI_LD_0_CO = cci_values["0 graden extensie with co-contraction"]["CCI_LD_CO"]

CCI_TM_90 = cci_values["90 graden extensie without co-contraction"]["CCI_TM"]
CCI_TM_90_CO = cci_values["90 graden extensie with co-contraction"]["CCI_TM_CO"]
CCI_LD_90 = cci_values["90 graden extensie without co-contraction"]["CCI_LD"]
CCI_LD_90_CO = cci_values["90 graden extensie with co-contraction"]["CCI_LD_CO"]

CCI_TM_180 = cci_values["180 graden extensie without co-contraction"]["CCI_TM"]
CCI_TM_180_CO = cci_values["180 graden extensie with co-contraction"]["CCI_TM_CO"]
CCI_LD_180 = cci_values["180 graden extensie without co-contraction"]["CCI_LD"]
CCI_LD_180_CO = cci_values["180 graden extensie with co-contraction"]["CCI_LD_CO"]

# gemiddelde CCI per meting
CCI_TM = (CCI_TM_180 + CCI_TM_90 + CCI_TM_0)/3
CCI_LD = (CCI_LD_180 + CCI_LD_90 + CCI_LD_0)/3
CCI_TM_CO = (CCI_TM_180_CO + CCI_TM_90_CO + CCI_TM_0_CO)/3
CCI_LD_CO = (CCI_LD_180_CO + CCI_LD_90_CO + CCI_LD_0_CO)/3

percentage_LD_norm = 0.6
percentage_TM_norm = 0.6

# CCI thresholds zetten
threshold_CCI_TM = CCI_TM * percentage_TM_norm + CCI_TM_CO * (1- percentage_TM_norm)
threshold_CCI_LD = CCI_LD * percentage_LD_norm + CCI_LD_CO * (1- percentage_LD_norm)

# DM threshold
DM_180 = cci_values["180 graden extensie without co-contraction"]["mean_DM"]
DM_90 = cci_values["90 graden extensie without co-contraction"]["mean_DM"]

threshold_DM = (DM_180 + DM_90)/10 # 1/5 van de gemiddelde DM waardes in de 2 oefeningen

# PRE-PROCESSING FUNCTIES -------------------------------------------------------------------------------
#Lowpass-filter functie 
def lowpass(data, cutoff, fs, poles=5):
    sos = scipy.signal.butter(N=poles, Wn=cutoff, btype='low', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)

# Root Mean Square functie
def moving_rms(data, window_ms, fs):
    w = int(window_ms/1000 * fs) or 1
    sq = data**2
    kernel = np.ones(w) / w
    rms = np.empty_like(data)
    for ch in range(data.shape[1]):
        ms = np.convolve(sq[:, ch], kernel, mode='same')
        rms[:, ch] = np.sqrt(ms)
    return rms

# Highpass-filter functie
def highpass(data, cutoff, fs, poles=5):
    sos = scipy.signal.butter(N=poles,Wn=cutoff,btype='high', fs=fs,output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)

# CCI bereken functie
def compute_cci(emg1, emg2):
    return (emg1 / (emg2 + 1e-8)) * (emg1 + emg2)

# PLOTTEN PER BESTAND --------------------------------------------------------------------------------------------
cocontractie_detecties_concat = []
cocontractie_aannames_concat = []

for file in selected_files:
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        data = np.array([[float(val) for val in row] for row in reader])

    # Preprocessing
    emg_hp = highpass(data, cutoff=10, fs=1000)
    emg_rms = moving_rms(emg_hp, window_ms=50, fs=1000)
    emg_lp = lowpass(emg_rms, cutoff=450, fs=1000)

    # Normaliseren 
    filtered_emg = np.empty_like(emg_lp)
    filtered_emg[:, 0] = emg_lp[:, 0] / mvic_DM
    filtered_emg[:, 1] = emg_lp[:, 1] / mvic_LD
    filtered_emg[:, 2] = emg_lp[:, 2] / mvic_TM

    CCI_LD = compute_cci(filtered_emg[:, 1], filtered_emg[:, 0])
    CCI_TM = compute_cci(filtered_emg[:, 2], filtered_emg[:, 0])
    
     # Drempel deltoid contractie 
    initial_mask = filtered_emg[:, 0] > threshold_DM

    # Minimale duur
    labeled_on, n_on = label(initial_mask)
    true_mask = np.zeros_like(initial_mask, dtype=bool)
    for seg in range(1, n_on+1):
        idx = (labeled_on == seg)
        if idx.sum() >= 200:
            true_mask[idx] = True

    # Korte onderbrekingen (False-segmenten) opvullen
    inv_mask = ~true_mask
    labeled_off, n_off = label(inv_mask)
    final_mask = true_mask.copy()
    for seg in range(1, n_off+1):
        idx = (labeled_off == seg)
        if idx.sum() < 100:
            final_mask[idx] = True

    delt_active = final_mask

    # Maskers voor co-contractie onder CCI-drempels
    mask_tm = (CCI_TM < threshold_CCI_TM) & delt_active
    mask_ld = (CCI_LD < threshold_CCI_LD) & delt_active

    # Sliding-window count
    count_tm = np.convolve(mask_tm.astype(int), np.ones(window, dtype=int), mode='same')
    count_ld = np.convolve(mask_ld.astype(int), np.ones(window, dtype=int), mode='same')

    # Bepaal waar binnen enige window de drempel_samples wordt overschreden
    beep_mask = (count_tm >= threshold_samples) | (count_ld >= threshold_samples)

    # Draai om naar labels: 0 = piep, 1 = geen piep
    cocontractie_detectie = np.where(beep_mask, 0, 1)
    cocontractie_detecties_concat.append(cocontractie_detectie)

    # Haal de file-label (0 of 1) en maak er een vector van dezelfde lengte van
    y_true = labels_per_file[file]
    true_vector = np.full_like(cocontractie_detectie, fill_value=y_true)
    cocontractie_aannames_concat.append(true_vector)

    # Parameters voor plotting
    N = filtered_emg.shape[0]
    t = np.arange(0, N * interval, interval)

    # Plot voor elk bestand
    fig, axs = plt.subplots(n_channels + 2, 1, figsize=(18, 10), sharex=True)
    spiertitels = ['Deltoid', 'Latissimus dorsi', 'Teres major']

    for i in range(n_channels):
        axs[i].plot(t, filtered_emg[:, i])
        axs[i].set_ylabel(header[i])
        axs[i].set_ylim(0, 0.5)
        axs[i].grid(True)
        axs[i].set_title(spiertitels[i], fontsize=10)
    axs[0].fill_between(t, 0, 1, where=delt_active, facecolor='pink', alpha=0.3, interpolate=True)
    axs[0].axhline(y=threshold_DM, color='orange', linestyle='--', label='Threshold')

    axs[2].fill_between(t, 0, 25, where=beep_mask, facecolor='red', alpha=0.2)
        
    axs[n_channels].plot(t, CCI_LD, color='orange')
    axs[n_channels].set_ylabel("CCI_LD")
    axs[n_channels].set_ylim(0, 25)
    axs[n_channels].grid(True)
    axs[n_channels].set_title('CCI Latissimus dorsi', fontsize=10)
    
    axs[n_channels + 1].plot(t, CCI_TM, color='purple')
    axs[n_channels + 1].set_ylabel("CCI_TM")
    axs[n_channels + 1].set_ylim(0, 25)
    axs[n_channels + 1].grid(True)
    axs[n_channels + 1].set_title('CCI Teres major', fontsize=10)
 
    axs[-1].set_xlabel("Tijd (s)")
    filename_only = os.path.basename(file)
    fig.suptitle(f"EMG Signalen en Co-Contractie Ratio's: {filename_only}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# STATISTIEKEN BEREKENEN------------------------------------------------------------------------------------
arr_cocontractie_detectie = np.concatenate(cocontractie_detecties_concat)
arr_cocontractie_aanname = np.concatenate(cocontractie_aannames_concat)

# Confusion matrix
cm = confusion_matrix(arr_cocontractie_aanname, arr_cocontractie_detectie, labels=[0,1])
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")

# SAVE VERIFICATION RESULTS---------------------------------------------------------------------------------
results = {
    "participant": ptnr,
    "sensitivity": sensitivity,
    "specificity": specificity
}

if save and save.lower() == "ja":
    results_file = os.path.join(os.getcwd(), "verification_results3.json")

    # Laad bestaande resultaten als die er zijn, anders begin opnieuw.
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as rf:
                all_results = json.load(rf)
        except json.JSONDecodeError:
            all_results = []
    else:
        all_results = []

    all_results.append(results)

    with open(results_file, 'w') as wf:
        json.dump(all_results, wf, indent=4)

    print(f"✅ Verification results saved to {results_file}")
