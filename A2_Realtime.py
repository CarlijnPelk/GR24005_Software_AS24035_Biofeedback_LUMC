#%% IMPORTEREN PACKAGES ---------------------------------------------------------------------------------------
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
import csv
import os
import json
import scipy.signal
import threading
from collections import deque
from scipy.ndimage import label
import sounddevice as sd

#%% POP-UP ----------------------------------------------------------------------------------------------------
root = tk.Tk() 
root.withdraw()

ptnr = simpledialog.askstring("Participant Info", "Participant number:")
exercise = simpledialog.askstring("Exercise", "Exercise 1 = iso, \n Exercise 2 = muur, \n Exercise 3 = haar, \n Exercise 4 = gor:")
bf = simpledialog.askstring("Biofeedback", "With biofeedback = 1, \n without biofeedback = 2:")

while bf not in ("1", "2"):
    bf = simpledialog.askstring("Biofeedback", "Invalid Biofeedback. \n\n Enter: 1 or 2:")

root.deiconify()

#%% Parameters -----------------------------------------------------------------------------------------------
device = "Dev2"
channels = ["4", "6", "7"]
sample_rate = 1000 # Hz
n_channels = len(channels)
window = 500
threshold_samples = int(0.55 * window)
alert_mask = deque(maxlen=50000) 

# Geluidsinstellingen
fs = 44100
freq = 440.0  # Hz
tone_stream = None  

# Begin status
running = False
saving = False

# Data lijsten
data_list = []  
plot_queue = deque(maxlen=1)                
plot_queue.append(np.zeros((500, n_channels)))

#%% LAAD MVIC FILE --------------------------------------------------------------------------------------------
mvic_file = f"mvic_{ptnr}.json"
mvic_dir = os.path.join(os.getcwd(), "mvic")
load_path = os.path.join(mvic_dir, mvic_file)
mvic_data = {}
if os.path.exists(load_path):
    with open(load_path, 'r') as f:
        mvic_data = json.load(f)
    print("Loaded MVIC values:", mvic_data)
else:
    print(f"❌ No MVIC file found for participant {ptnr}. Using MVIC=1 for all channels.")
mvic_DM = mvic_data.get("abduction", {}).get("mvic_DM", 1)
mvic_LD = mvic_data.get("adduction", {}).get("mvic_LD", 1)
mvic_TM = mvic_data.get("adduction", {}).get("mvic_TM", 1)

#%% CSV FILE AANMAKEN ----------------------------------------------------------------------------------------
timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
filename = f"EMG_ptnr{ptnr}_ex{exercise}_bf{bf}_{timestamp}.csv"
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"EMG_Output_ptnr{ptnr}")
os.makedirs(output_dir, exist_ok=True)
filepath = os.path.join(output_dir, filename)

#%% LAAD CCI FILE --------------------------------------------------------------------------------------------
# Path laden
folder = "cci measurements"
fname  = f"cci_{ptnr}.json"
path   = os.path.join(folder, fname)

with open(path, 'r') as f:
    cci_values = json.load(f)
CCI_TM_0 = cci_values["0 graden extensie without co-contraction"]["CCI_TM"]
CCI_TM_0_CO = cci_values["0 graden extensie with co-contraction"]["CCI_TM_CO"]
CCI_LD_0 = cci_values["0 graden extensie without co-contraction"]["CCI_LD"]
CCI_LD_0_CO = cci_values["0 graden extensie with co-contraction"]["CCI_LD_CO"]

CCI_TM_180 = cci_values["180 graden extensie without co-contraction"]["CCI_TM"]
CCI_LD_180 = cci_values["180 graden extensie without co-contraction"]["CCI_LD"]
CCI_TM_CO_180 = cci_values["180 graden extensie with co-contraction"]["CCI_TM_CO"]
CCI_LD_CO_180 = cci_values["180 graden extensie with co-contraction"]["CCI_LD_CO"]

CCI_TM_90 = cci_values["90 graden extensie without co-contraction"]["CCI_TM"]
CCI_LD_90 = cci_values["90 graden extensie without co-contraction"]["CCI_LD"]
CCI_TM_CO_90 = cci_values["90 graden extensie with co-contraction"]["CCI_TM_CO"]
CCI_LD_CO_90 = cci_values["90 graden extensie with co-contraction"]["CCI_LD_CO"]

DM_180 = cci_values["180 graden extensie without co-contraction"]["mean_DM"]
DM_90 = cci_values["90 graden extensie without co-contraction"]["mean_DM"]

#%% THRESHOLDS BEPALEN --------------------------------------------------------------------------------------
# CCI thresholds
CCI_TM = (CCI_TM_180 + CCI_TM_90 + CCI_TM_0)/3
CCI_LD = (CCI_LD_180 + CCI_LD_90 + CCI_LD_0)/3
CCI_TM_CO = (CCI_TM_CO_180 + CCI_TM_CO_90 + CCI_TM_0_CO)/3
CCI_LD_CO = (CCI_LD_CO_180 + CCI_LD_CO_90 + CCI_LD_0_CO)/3

percentage_LD = 0.6
percentage_TM = 0.6

threshold_CCI_TM = CCI_TM * percentage_TM + CCI_TM_CO * (1- percentage_TM)
threshold_CCI_LD = CCI_LD * percentage_LD + CCI_LD_CO * (1- percentage_LD)

# DM threshold
threshold_DM = (DM_180 + DM_90)/10

#%% PRE-PROCESSING FUNCTIES-----------------------------------------------------------------------------------
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

#%% CCI FUNCTIE ------------------------------------------------------------------------------------------------
def compute_cci(emg1, emg2):
    return (emg1 / (emg2 + 1e-8)) * (emg1 + emg2)

#%% GELUID FUNCTIE ---------------------------------------------------------------------------------------------
def sound_callback(outdata, frames, time, status):
    t = (np.arange(frames) + sound_callback.t0) / fs  # seconds
    sine_wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
    outdata[:] = np.column_stack([sine_wave, sine_wave])
    sound_callback.t0 += frames

#%% PLOTTEN ----------------------------------------------------------------------------------------------------
# Plot setup
plt.ion()
fig, axs = plt.subplots(n_channels+2, 1, figsize=(10, 6))
lines = [axs[i].plot(np.zeros(500))[0] for i in range(n_channels+2)]
emg_titles = ['Deltoid', 'Latissimus dorsi', 'Teres major']
for i, ax in enumerate(axs[:n_channels]):
    ax.set_ylabel("Amplitude EMG", fontsize=10)
    ax.set_title(emg_titles[i], fontsize=10)
    ax.set_xlim(0, 500)
axs[-1].set_xlabel("Sample")

plt.subplots_adjust(hspace=0.6, top=0.90)

fig.suptitle(f"EMG Signalen en Co-Contractie Ratio's: {filename}", fontsize=14)

y_ax_min = 0
prev_y_ax_max = [0.1] * n_channels
prev_y_ax_max_cci_tm = 0.1
prev_y_ax_max_cci_ld = 0.1

plt.show(block=False)

plot_after_id = None

# Realtime plotten
def update_plot():
    global prev_y_ax_max_cci_tm, prev_y_ax_max, prev_y_ax_max_cci_ld, plot_after_id
    try:
        plot_after_id = root.after(50, update_plot)

        if plot_queue:
            emg_block = plot_queue.popleft()
            for i in range(n_channels):
                lines[i].set_ydata(emg_block[:, i])
                lines[i].set_color('#FF1493')
                chunk_max = emg_block[:, i].max()
                if chunk_max > prev_y_ax_max[i] and chunk_max < 2:
                    prev_y_ax_max[i] = chunk_max
                margin = 0.1 * (prev_y_ax_max[i] - y_ax_min)
                axs[i].set_ylim(y_ax_min - margin, prev_y_ax_max[i] + margin)
            
            axs[0].axhline(y=threshold_DM, color='black', linestyle='--', label='Threshold')

            # CCI berekenen om te plotten
            cci_ld = compute_cci(emg_block[:, 1], emg_block[:, 0])  
            cci_tm = compute_cci(emg_block[:, 2], emg_block[:, 0])  

            lines[3].set_ydata(cci_ld)
            lines[3].set_color("blue")
            lines[4].set_ydata(cci_tm)
            lines[4].set_color("blue")

            # x waardes overeen laten komen met wanneer de piep afgaat
            if len(alert_mask) >= emg_block.shape[0]:
                current_alert_segment = np.array(list(alert_mask)[-emg_block.shape[0]:])
            else:
                current_alert_segment = np.zeros(emg_block.shape[0], dtype=int)

            x_vals = np.arange(len(current_alert_segment))               

            # instellingen voor latissimus dorsi plot
            axs[3].set_ylim(y_ax_min, threshold_CCI_LD * 6)
            axs[3].axhline(y=threshold_CCI_LD, color='black', linestyle='--', label='Threshold')
            
            axs[3].set_ylabel("Amplitude CCI", fontsize=10)
            axs[3].set_title("CCI - Latissimus dorsi", fontsize=10)

            # rood als de piep afgaat
            for coll in axs[3].collections:
                coll.remove()
            axs[3].fill_between(
                x_vals,
                y_ax_min,
                threshold_CCI_LD * 6,
                where=current_alert_segment.astype(bool),
                facecolor='darkred',
                alpha=0.3,
                interpolate=True)
            
            # instellingen voor teres major plot
            axs[4].set_ylim(y_ax_min, threshold_CCI_TM * 6)     
            axs[4].axhline(y=threshold_CCI_TM, color='black', linestyle='--', label='Threshold')       

            axs[4].set_ylabel("Amplitude CCI")
            axs[4].set_title("CCI - Teres major", fontsize=10)
       
            # rood als de piep afgaat
            for coll in axs[4].collections:
                coll.remove()
            axs[4].fill_between(
                x_vals,
                y_ax_min,
                threshold_CCI_TM * 6,
                where=current_alert_segment.astype(bool),
                facecolor='darkred',
                alpha=0.3,
                interpolate=True
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

    except tk.TclError: 
        return

plot_after_id = root.after(50, update_plot)

#%% DATA ACQUISITIE ----------------------------------------------------------------------------------------
def acquisition_loop():
    global running, saving, data_list
    try:
        with nidaqmx.Task() as task:
            for ch in channels:
                task.ai_channels.add_ai_voltage_chan(f"{device}/ai{ch}", terminal_config=TerminalConfiguration.DIFF)
            task.timing.cfg_samp_clk_timing(rate=sample_rate, sample_mode=AcquisitionType.CONTINUOUS)
            task.start()
            print("⏳ DAQ acquisition running (preview).")

            while running:
                values = task.read(number_of_samples_per_channel=500)  
                values = np.array(values).T 

                # Pre‐processing 
                emg_hp = highpass(values, cutoff=10, fs=sample_rate)
                emg_rms = moving_rms(emg_hp, window_ms=50, fs=sample_rate)
                emg_lp = lowpass(emg_rms, cutoff=450, fs=sample_rate)

                # Normaliseren 
                filtered_emg = np.empty_like(emg_lp)
                filtered_emg[:, 0] = emg_lp[:, 0] / mvic_DM
                filtered_emg[:, 1] = emg_lp[:, 1] / mvic_LD
                filtered_emg[:, 2] = emg_lp[:, 2] / mvic_TM

                CCI_LD = compute_cci(filtered_emg[:, 1], filtered_emg[:, 0])
                CCI_TM = compute_cci(filtered_emg[:, 2], filtered_emg[:, 0])

                if bf == "1":                   
                    initial_mask = filtered_emg[:, 0] > threshold_DM

                    # Forceer minimale duur van True-segmenten
                    labeled_array, num_features = label(initial_mask)
                    true_mask = np.zeros_like(initial_mask, dtype=bool)
                    for i in range(1, num_features + 1):
                        segment = (labeled_array == i)
                        if np.sum(segment) >= 200:
                            true_mask[segment] = True

                    # Forceer minimale duur van False-segmenten 
                    inverse_mask = ~true_mask
                    labeled_inv, num_inv = label(inverse_mask)
                    final_mask = true_mask.copy()
                    for i in range(1, num_inv + 1):
                        segment = (labeled_inv == i)
                        if np.sum(segment) < 100:
                            final_mask[segment] = True 

                    delt_active = final_mask

                    # Maskers voor co-contractie onder drempel en als delt actief is
                    mask_tm = (CCI_TM < threshold_CCI_TM) & delt_active
                    mask_ld = (CCI_LD < threshold_CCI_LD) & delt_active

                    # Sliding window over maskers
                    count_tm = np.convolve(mask_tm.astype(int), np.ones(window, dtype=int), mode='same')
                    count_ld = np.convolve(mask_ld.astype(int), np.ones(window, dtype=int), mode='same')

                    # Check of binnen laatste sample het aantal overschrijdingen boven de drempel komt
                    if np.any(count_tm[-window:] >= threshold_samples) or np.any(count_ld[-window:] >= threshold_samples):
                        start_tone()
                        alert_vector = np.ones(filtered_emg.shape[0], dtype=int)  
                    else:
                        stop_tone()
                        alert_vector = np.zeros(filtered_emg.shape[0], dtype=int)

                    alert_mask.extend(alert_vector)

                # Toevoegen aan queue
                plot_queue.append(filtered_emg)

                # If saving==True, sla de RUWE data op in data_list
                if saving:
                    data_list.extend(values.tolist())

            # Stop als runnen stopt
            task.stop()
            print("⏹ DAQ acquisition stopped.")

    except Exception as e:
        print("‼️ Acquisition thread error:", e)
        running = False
        saving = False

#%% KNOPPEN --------------------------------------------------------------------------------------------------
# Knoppen definiëren
def start_tone():
    global tone_stream
    if tone_stream is None:
        sound_callback.t0 = 0
        tone_stream = sd.OutputStream(
            samplerate=fs,
            channels=2,        
            dtype='float32',
            callback=sound_callback
        )
        tone_stream.start()

def stop_tone():
    global tone_stream
    if tone_stream is not None:
        tone_stream.stop()
        tone_stream.close()
        tone_stream = None

def start_preview():
    global running, acq_thread
    if not running:
        running = True
        acq_thread = threading.Thread(target=acquisition_loop, daemon=True)
        acq_thread.start()
        preview_btn.config(state="disabled")
        save_btn.config(state="normal")
        stop_btn.config(state="normal")

def start_saving():
    global saving
    if running and not saving:
        saving = True
        save_btn.config(state="disabled")
        print("💾 Data saving has started.")

def stop_all():
    global running, saving
    if running:
        saving = False
        running = False
        stop_btn.config(state="disabled")
        save_btn.config(state="disabled")
        preview_btn.config(state="normal")
        acq_thread.join()  
        if plot_after_id is not None:
            try:
                root.after_cancel(plot_after_id)
            except Exception:
                pass         
        save_and_exit()

def save_and_exit():
    if data_list:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([f"EMG_Channel_{i+1}" for i in range(n_channels)])
            writer.writerows(data_list)
        print(f"✅ Data saved to: {filepath}")
    else:
        print("⚠️ No data was recorded (saving was never activated).")
    root.destroy()

# Knoppen maken in pop-up
ctrl = tk.Frame(root)
ctrl.pack(pady=5)

preview_btn = tk.Button(ctrl, text="Preview", command=start_preview)
preview_btn.pack(side="left", padx=5)

save_btn = tk.Button(ctrl, text="Start Saving", command=start_saving, state="disabled")
save_btn.pack(side="left", padx=5)

stop_btn = tk.Button(ctrl, text="Stop", command=stop_all, state="disabled")
stop_btn.pack(side="left", padx=5)

root.mainloop()