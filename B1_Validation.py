#%%
# IMPORTEREN PACKAGES-------------------------------------------------------------------------------------------
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import json
from scipy.ndimage import label
from scipy.stats import wilcoxon
#%% POP-UP ----------------------------------------------------------------------------------------------------
interval = 0.001
n_channels = 3

# Tkinter setup
root = tk.Tk()
root.withdraw()

#%% BESTANDEN SELECTEREN -------------------------------------------------------
selected_files = []
participant_per_file = {}
labels_per_file = {}

while True:
    ptnr = simpledialog.askstring("Participant Info", "Participant nummer:")
    if not ptnr:
        messagebox.showerror("Fout", "Geen participantnummer ingevoerd.")
        break

    mapnaam = f"EMG_Output_ptnr{ptnr}"
    huidige_map = os.getcwd()
    pad_naar_map = os.path.join(huidige_map, mapnaam)

    if not os.path.isdir(pad_naar_map):
        messagebox.showerror("Fout", f"\u274c De map '{mapnaam}' bestaat niet.")
        continue

    files = filedialog.askopenfilenames(
        initialdir=pad_naar_map,
        title=f"Kies één of meerdere CSV-bestand van participant {ptnr}",
        filetypes=[("CSV bestanden", "*.csv")]
    )

    if not files:
        messagebox.showwarning("Geen bestand", "Geen bestand(en) geselecteerd.")
        continue

    for file in files:
        bestand = os.path.basename(file)

        if "bf2" in bestand:
            labels_per_file[file] = 0        # zonder biofeedback
        else:
            labels_per_file[file] = 1  
        participant_per_file[file] = ptnr        # met biofeedback

    selected_files.extend(files)
    
    opnieuw = messagebox.askyesno("Bestand toevoegen?", "Wil je nog een bestand toevoegen?")
    if not opnieuw:
        break

#%% PRE-PROCESSING FUNCTIES -----------------------------------------------------------------------------------
# Lowpass-filter functie 
def lowpass(data, cutoff, fs, poles=5):
    sos = scipy.signal.butter(N=poles, Wn=cutoff, btype='low', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)

# Moving RMS filter
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
    sos = scipy.signal.butter(N=poles, Wn=cutoff, btype='high', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)

#%% CCI FUNCTIE----------------------------------------------------------------------------------------------
def compute_cci(emg1, emg2):
    return (emg1 / (emg2 + 1e-8)) * (emg1 + emg2)

#%% RESULTATEN ----------------------------------------------------------------------------------------------
mean_CCI_LD_per_ptnr = {}
mean_CCI_TM_per_ptnr = {}

# Verwerking per bestand ------------------------------------------------------------------------------------
for file in selected_files:
    ptnr = participant_per_file[file]

    # CCI laden
    with open(os.path.join("cci measurements", f"cci_{ptnr}.json"), 'r') as f:
        cci_values = json.load(f)

    CCI_TM = np.mean([
        cci_values[hoek]["CCI_TM"] for hoek in [
            "0 graden extensie without co-contraction",
            "90 graden extensie without co-contraction",
            "180 graden extensie without co-contraction"]])
    CCI_LD = np.mean([
        cci_values[hoek]["CCI_LD"] for hoek in [
            "0 graden extensie without co-contraction",
            "90 graden extensie without co-contraction",
            "180 graden extensie without co-contraction"]])

    CCI_TM_CO = np.mean([
        cci_values[hoek]["CCI_TM_CO"] for hoek in [
            "0 graden extensie with co-contraction",
            "90 graden extensie with co-contraction",
            "180 graden extensie with co-contraction"]])
    CCI_LD_CO = np.mean([
        cci_values[hoek]["CCI_LD_CO"] for hoek in [
            "0 graden extensie with co-contraction",
            "90 graden extensie with co-contraction",
            "180 graden extensie with co-contraction"]])

    percentage_LD_norm = 0.6
    percentage_TM_norm = 0.6
    threshold_CCI_TM = CCI_TM * percentage_TM_norm + CCI_TM_CO * (1 - percentage_TM_norm)
    threshold_CCI_LD = CCI_LD * percentage_LD_norm + CCI_LD_CO * (1 - percentage_LD_norm)

    # DM threshold berekenen
    DM_180 = cci_values["180 graden extensie without co-contraction"]["mean_DM"]
    DM_90 = cci_values["90 graden extensie without co-contraction"]["mean_DM"]
    threshold_DM = (DM_180 + DM_90) / 10

    # MVIC laden
    mvic_file = os.path.join("mvic", f"mvic_{ptnr}.json")
    with open(mvic_file, 'r') as f:
        mvic_data = json.load(f)

    mvic_DM = mvic_data.get("abduction", {}).get("mvic_DM", 1)
    mvic_LD = mvic_data.get("adduction", {}).get("mvic_LD", 1)
    mvic_TM = mvic_data.get("adduction", {}).get("mvic_TM", 1)

    # Data laden
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

    CCI_LD_array = compute_cci(filtered_emg[:, 1], filtered_emg[:, 0])
    CCI_TM_array = compute_cci(filtered_emg[:, 2], filtered_emg[:, 0])

    initial_mask = filtered_emg[:, 0] > threshold_DM
    labeled_array, num_features = label(initial_mask)
    true_mask = np.zeros_like(initial_mask, dtype=bool)
    for i in range(1, num_features + 1):
        segment = (labeled_array == i)
        if np.sum(segment) >= 200:
            true_mask[segment] = True

    inverse_mask = ~true_mask
    labeled_inv, num_inv = label(inverse_mask)
    final_mask = true_mask.copy()
    for i in range(1, num_inv + 1):
        segment = (labeled_inv == i)
        if np.sum(segment) < 100:
            final_mask[segment] = True

    delt_active = final_mask
    mask_fb = (CCI_TM_array < threshold_CCI_TM) & delt_active
    mask_fbld = (CCI_LD_array < threshold_CCI_LD) & delt_active

    N = filtered_emg.shape[0]
    t = np.arange(0, N * interval, interval)
    window = 500
    threshold_samples = int(0.55 * window)
    beep_given = np.zeros(N, dtype=bool)

    for start in range(0, N, 500):
        block = slice(start, min(start + 500, N))
        count_tm_block = np.convolve(mask_fb[block].astype(int), np.ones(window, dtype=int), mode='same')
        count_ld_block = np.convolve(mask_fbld[block].astype(int), np.ones(window, dtype=int), mode='same')
        if np.any(count_tm_block[-window:] >= threshold_samples) or np.any(count_ld_block[-window:] >= threshold_samples):
            beep_given[block] = True

    CCI_LD_array_verkort = CCI_LD_array[1000:-1000]
    CCI_TM_array_verkort = CCI_TM_array[1000:-1000]
    CCI_LD_above_threshold = CCI_LD_array_verkort > threshold_CCI_LD
    CCI_TM_above_threshold = CCI_TM_array_verkort > threshold_CCI_TM
    percentage_CCI_LD_above = np.sum(CCI_LD_above_threshold) / len(CCI_LD_above_threshold) 
    percentage_CCI_TM_above = np.sum(CCI_TM_above_threshold) / len(CCI_TM_above_threshold) 
    mean_CCI_LD = percentage_CCI_LD_above
    mean_CCI_TM = percentage_CCI_TM_above
    print(f"Percentage CCI_LD boven threshold: {percentage_CCI_LD_above:.3f}")
    print(f"Percentage CCI_TM boven threshold: {percentage_CCI_TM_above:.3f}")
   
    biofeedback_label = labels_per_file[file] == 1  # True = biofeedback, False = geen biofeedback

    if ptnr not in mean_CCI_LD_per_ptnr:
        mean_CCI_LD_per_ptnr[ptnr] = {True: [], False: []}
        mean_CCI_TM_per_ptnr[ptnr] = {True: [], False: []}

    mean_CCI_LD_per_ptnr[ptnr][biofeedback_label].append(mean_CCI_LD)
    mean_CCI_TM_per_ptnr[ptnr][biofeedback_label].append(mean_CCI_TM)

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

    axs[n_channels].plot(t, CCI_LD_array, color='orange')
    axs[n_channels].set_ylabel("CCI_LD")
    axs[n_channels].set_ylim(0, 25)
    axs[n_channels].grid(True)
    axs[n_channels].set_title('CCI Latissimus dorsi', fontsize=10)
    axs[n_channels].fill_between(t, 0, 25, where=mask_fbld, facecolor='orange', alpha=0.3, interpolate=True)

    axs[n_channels + 1].plot(t, CCI_TM_array, color='purple')
    axs[n_channels + 1].set_ylabel("CCI_TM")
    axs[n_channels + 1].set_ylim(0, 25)
    axs[n_channels + 1].grid(True)
    axs[n_channels + 1].set_title('CCI Teres major', fontsize=10)
    axs[n_channels + 1].fill_between(t, 0, 25, where=mask_fb, facecolor='purple', alpha=0.3, interpolate=True)
    axs[2].fill_between(t, 0, 0.5, where=beep_given, facecolor='darkred', alpha=0.3, interpolate=True)

    axs[-1].set_xlabel("Tijd (s)")
    fig.suptitle(f"EMG Signalen en Co-Contractie Ratio's: {os.path.basename(file)}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    
#%% OVERZICHT -----------------------------------------------------------------------------------------------

mean_LD_without_bf = []
mean_LD_with_bf = []
mean_TM_without_bf = []
mean_TM_with_bf = []

for ptnr in sorted(mean_CCI_LD_per_ptnr.keys()):  
    without_bf_vals_LD = mean_CCI_LD_per_ptnr[ptnr][False]
    with_bf_vals_LD = mean_CCI_LD_per_ptnr[ptnr][True]
    
    # Gemiddelde per participant per conditie, alleen als er waarden zijn
    if without_bf_vals_LD:
        mean_LD_without_bf.append(np.mean(without_bf_vals_LD))
    else:
        mean_LD_without_bf.append(np.nan)  
    
    if with_bf_vals_LD:
        mean_LD_with_bf.append(np.mean(with_bf_vals_LD))
    else:
        mean_LD_with_bf.append(np.nan)  

print("Gemiddelde percentage co-contractie LD per participant ZONDER biofeedback:", mean_LD_without_bf)
print("Gemiddelde percentage co-contractie LD per participant MET biofeedback:", mean_LD_with_bf)

for ptnr in sorted(mean_CCI_TM_per_ptnr.keys()):  
    without_bf_vals_TM = mean_CCI_TM_per_ptnr[ptnr][False]
    with_bf_vals_TM = mean_CCI_TM_per_ptnr[ptnr][True]
    
    # Gemiddelde per participant per conditie, alleen als er waarden zijn
    if without_bf_vals_TM:
        mean_TM_without_bf.append(np.mean(without_bf_vals_TM))
    else:
        mean_TM_without_bf.append(np.nan)  

    if with_bf_vals_TM:
        mean_TM_with_bf.append(np.mean(with_bf_vals_TM))
    else:
        mean_TM_with_bf.append(np.nan)  

print("Gemiddelde percentage co-contractie TM per participant ZONDER biofeedback:", mean_TM_without_bf)
print("Gemiddelde percentage co-contractie TM per participant MET biofeedback:", mean_TM_with_bf)

mean_LD_no_bf_total = np.nanmean(mean_LD_without_bf)
std_LD_no_bf_total = np.nanstd(mean_LD_without_bf)
mean_LD_with_bf_total = np.nanmean(mean_LD_with_bf)
std_LD_with_bf_total = np.nanstd(mean_LD_with_bf)
mean_TM_no_bf_total = np.nanmean(mean_TM_without_bf)
std_TM_no_bf_total = np.nanstd(mean_TM_without_bf)
mean_TM_with_bf_total = np.nanmean(mean_TM_with_bf)
std_TM_with_bf_total = np.nanstd(mean_TM_with_bf)
median_LD_no_bf_total = np.nanmedian(mean_LD_without_bf)
median_LD_with_bf_total = np.nanmedian(mean_LD_with_bf)
print(f"\nTotale gemiddelde percentage co-contractie LD zonder biofeedback, mediaan: {mean_LD_no_bf_total:.3f} (SD = {std_LD_no_bf_total:.3f}, Mediaan = {median_LD_no_bf_total:.3f})")
print(f"Totale gemiddelde percentage co-contractie LD met biofeedback, mediaan:    {mean_LD_with_bf_total:.3f} (SD = {std_LD_with_bf_total:.3f}, Mediaan = {median_LD_with_bf_total:.3f})")
median_TM_no_bf_total = np.nanmedian(mean_TM_without_bf)
median_TM_with_bf_total = np.nanmedian(mean_TM_with_bf)

print(f"\nTotale gemiddelde percentage co-contractie TM zonder biofeedback, mediaan: {mean_TM_no_bf_total:.3f} (SD = {std_TM_no_bf_total:.3f}, Mediaan = {median_TM_no_bf_total:.3f})")
print(f"Totale gemiddelde percentage co-contractie TM met biofeedback, mediaan:    {mean_TM_with_bf_total:.3f} (SD = {std_TM_with_bf_total:.3f}, Mediaan = {median_TM_with_bf_total:.3f})")


##% WILCOXON SIGNED RANK TEST-----------------------------------------------------------------------------------

stat, p_value = wilcoxon(mean_LD_with_bf, mean_LD_without_bf, alternative='greater')

print("\n--- Wilcoxon Signed-Rank Test LD ---")
print(f"Statistiek: {stat:.4f}")
print(f"P-waarde (H1: met biofeedback > zonder biofeedback): {p_value:.4f}")

if p_value < 0.05:
    print("✅ Significant verschil: Percentage co-contractie LD is hoger met biofeedback (p < 0.05).")
else:
    print("❌ Geen significant verschil (p >= 0.05).")

stat, p_value = wilcoxon(mean_TM_with_bf, mean_TM_without_bf, alternative='greater')

print("\n--- Wilcoxon Signed-Rank Test TM ---")
print(f"Statistiek: {stat:.4f}")
print(f"P-waarde (H1: met biofeedback > zonder biofeedback): {p_value:.4f}")

if p_value < 0.05:
    print("✅ Significant verschil: Percentage co-contractie TM is hoger met biofeedback (p < 0.05).")
else:
    print("❌ Geen significant verschil (p >= 0.05).")

# BOXPLOT ----------------------------------------------------------------------------------------------------

# CCI_LD boxplot
fig, ax = plt.subplots()
ax.boxplot([mean_LD_without_bf, mean_LD_with_bf], labels=['Zonder biofeedback', 'Met biofeedback'])
ax.set_title('CCI_LD per conditie')
ax.set_ylabel('Gemiddelde CCI_LD')
plt.grid(True)
plt.tight_layout()
plt.show()

# CCI_TM boxplot
fig, ax = plt.subplots()
ax.boxplot([mean_TM_without_bf, mean_TM_with_bf], labels=['Zonder biofeedback', 'Met biofeedback'])
ax.set_title('CCI_TM per conditie')
ax.set_ylabel('Gemiddelde CCI_TM')
plt.grid(True)
plt.tight_layout()
plt.show()
