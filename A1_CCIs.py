#%% IMPORTEREN PACKAGES -------------------------------------------------------------------------------------------
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
import os
import json
import scipy.signal
import threading

#%% POP-UP --------------------------------------------------------------------------------------------------------

root = tk.Tk()

# Participant nummer kiezen
ptnr   = simpledialog.askstring( "Participant Info",   "Participant number:", parent=root )

root.withdraw()

# Verschillende keuzes toevoegen
v = tk.IntVar(value=1)
cci_choice = {"180 graden extensie without co-contraction":1,
               "180 graden extensie with co-contraction":2,
               "90 graden extensie without co-contraction":3,
               "90 graden extensie with co-contraction":4,
               "0 graden extensie without co-contraction":5,
               "0 graden extensie with co-contraction":6,}
reverse_cci_choice = {val:key for key,val in cci_choice.items()}

frame = tk.Toplevel(root)
tk.Label(frame, text="Choose if the participants co-contracts:").pack(padx=20, pady=10)
for text, val in cci_choice.items():
    tk.Radiobutton(frame, text=text, variable=v, value=val).pack(anchor="w", padx=20)

def submit_choice():
    global cci_type
    cci_type = reverse_cci_choice[v.get()]
    print("You chose", cci_type)
    frame.destroy()
    root.quit()

tk.Button(frame, text="Submit", command=submit_choice).pack(pady=10)

root.mainloop()   
root.destroy()
print("✅ Questions completed. Continuing…")

#%% LAAD CCI FILE OF MAAK EEN NIEUWE -----------------------------------------------------------------------------
filename = f"cci_{ptnr}.json"

# Definieer foldernaam
folder_name = "cci measurements"

# Maak een folder als die niet bestaat
folder_path = os.path.join(os.getcwd(), folder_name)
os.makedirs(folder_path, exist_ok=True)

# Pad opslaan voor file binnen de folder
save_path = os.path.join(folder_path, filename)

if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        cci_values = json.load(f)
else:
    cci_values = {"participant": ptnr}

#%% PREPROCESSING FUNCTIES  -----------------------------------------------------------------------------------
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
    sos = scipy.signal.butter(N=poles,Wn=cutoff,btype='high', fs=fs,output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)

#%% LAAD MVIC FILE ----------------------------------------------------------------------------------------------
mvic_file = f"mvic_{ptnr}.json"
mvic_dir = os.path.join(os.getcwd(), "mvic")  
load_path = os.path.join(mvic_dir, mvic_file)

if os.path.exists(load_path):
    with open(load_path, 'r') as f:
        mvic_data = json.load(f)
    print("Loaded MVIC values:")
    print(mvic_data)
else:
    print(f"❌ No MVIC file found for participant {ptnr}")

mvic_DM = mvic_data.get("abduction", {}).get("mvic_DM",1)
mvic_LD = mvic_data.get("adduction", {}).get("mvic_LD",1)
mvic_TM = mvic_data.get("adduction", {}).get("mvic_TM",1)

#%% CCI FUNCTIE -------------------------------------------------------------------------------------------------
def compute_cci(emg1, emg2):
    return (emg1 / (emg2 + 1e-8)) * (emg1 + emg2)

#%% DATA ACQUISITIE ----------------------------------------------------------------------------------------------
device = "Dev2"
channels = ["4", "6", "7"]
sample_rate = 1000  # Hz
n_channels = len(channels)
data_list = []

# Start en stopscherm maken 
window = tk.Tk()
window.title("Start CCI meting")

label = tk.Label(window, text="Druk op de knop om CCI-meting te starten (5 seconden):", font=("Arial", 12))
label.pack(pady=10)

countdown_label = tk.Label(window, text="", font=("Arial", 24), fg="blue")
countdown_label.pack(pady=5)

# Aftellen functie 
def update_countdown(seconds_left):
    if seconds_left > 0:
        countdown_label.config(text=f"{seconds_left} sec")
        window.after(1000, update_countdown, seconds_left - 1)
    else:
        countdown_label.config(text="✅ Klaar!")

# Startfunctie 
def start_acquisition():
    start_btn.config(state=tk.DISABLED)
    update_countdown(5)

    def acquire():
        with nidaqmx.Task() as task:
            for ch in channels:
                task.ai_channels.add_ai_voltage_chan(
                    f"{device}/ai{ch}",
                    terminal_config=TerminalConfiguration.DIFF
                )
            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
            )

            task.start()
            print("⏳ Recording started for 5 seconds...")

            try:
                start_time = datetime.now()
                while (datetime.now() - start_time).total_seconds() < 5:
                    values = task.read(number_of_samples_per_channel=500)
                    values = np.array(values).T
                    for v in values:
                        data_list.append(v)

                print("🛑 Recording auto-stopped after 5 seconds.")

            except Exception as e:
                print(f"⚠️ Error during acquisition: {e}")
            finally:
                task.stop()
                window.quit()

    threading.Thread(target=acquire, daemon=True).start()

# Startknop
start_btn = tk.Button(window, text="▶ Start meting", font=("Arial", 14), bg="green", fg="white", command=start_acquisition)
start_btn.pack(pady=10)

window.mainloop()
window.destroy()

#%% CONVERTEREN DATA ------------------------------------------------------------------------------------------
emg_data = np.array(data_list) 
emg_hp = highpass(emg_data, cutoff=20, fs=1000)   # verwijder <20 Hz
emg_rms = moving_rms(emg_hp, window_ms=50, fs=1000) # rectificeren
emg_lp = lowpass(emg_rms, cutoff=450, fs=1000)   # verwijder >450 Hz

# Normalisatie naar MVIC
filtered_emg = np.empty_like(emg_lp)
filtered_emg[:, 0] = (emg_lp[:, 0] / mvic_DM)  
filtered_emg[:, 1] = (emg_lp[:, 1] / mvic_LD)    
filtered_emg[:, 2] = (emg_lp[:, 2] / mvic_TM) 

# Berekenen CCI
CCI_LD = compute_cci(filtered_emg[:, 1], filtered_emg[:, 0])  # LD & DM
CCI_TM = compute_cci(filtered_emg[:, 2], filtered_emg[:, 0])  # TM & DM

CCI_LD_mean = CCI_LD.mean()
CCI_TM_mean = CCI_TM.mean()

mean_DM = np.mean(filtered_emg[:, 0])

#%% OPSLAAN CCI -------------------------------------------------------------------------------------------
if cci_type == "180 graden extensie without co-contraction":
    print("\n CCI_LD mean, std:", CCI_LD_mean, ", ", CCI_LD.std())
    print("CCI_TM mean, std:", CCI_TM_mean, ", ", CCI_TM.std())
    print("mean DM:", mean_DM)
    cci_values["180 graden extensie without co-contraction"] = {
        "CCI_LD": float(CCI_LD_mean),
        "CCI_TM": float(CCI_TM_mean),
        "mean_DM": float(mean_DM)}
    
if cci_type == "180 graden extensie with co-contraction":
    print("\n CCI_LD_CO mean, std:", CCI_LD_mean, ", ", CCI_LD.std())
    print("CCI_TM_CO mean, std:", CCI_TM_mean, ", ", CCI_TM.std())
    cci_values["180 graden extensie with co-contraction"] = {
        "CCI_LD_CO": float(CCI_LD_mean),
        "CCI_TM_CO": float(CCI_TM_mean)}

if cci_type == "90 graden extensie without co-contraction":
    print("\n CCI_LD mean, std:", CCI_LD_mean, ", ", CCI_LD.std())
    print("CCI_TM mean, std:", CCI_TM_mean, ", ", CCI_TM.std())
    print("mean DM:", mean_DM)
    cci_values["90 graden extensie without co-contraction"] = {
        "CCI_LD": float(CCI_LD_mean),
        "CCI_TM": float(CCI_TM_mean),
        "mean_DM": float(mean_DM)}
    
if cci_type == "90 graden extensie with co-contraction":
    print("\n CCI_LD_CO mean, std:", CCI_LD_mean, ", ", CCI_LD.std())
    print("CCI_TM_CO mean, std:", CCI_TM_mean, ", ", CCI_TM.std())
    cci_values["90 graden extensie with co-contraction"] = {
        "CCI_LD_CO": float(CCI_LD_mean),
        "CCI_TM_CO": float(CCI_TM_mean)}
    
if cci_type == "0 graden extensie without co-contraction":
    print("\n CCI_LD mean, std:", CCI_LD_mean, ", ", CCI_LD.std())
    print("CCI_TM mean, std:", CCI_TM_mean, ", ", CCI_TM.std())
    cci_values["0 graden extensie without co-contraction"] = {
        "CCI_LD": float(CCI_LD_mean),
        "CCI_TM": float(CCI_TM_mean)}
    
if cci_type == "0 graden extensie with co-contraction":
    print("\n CCI_LD_CO mean, std:", CCI_LD_mean, ", ", CCI_LD.std())
    print("CCI_TM_CO mean, std:", CCI_TM_mean, ", ", CCI_TM.std())
    cci_values["0 graden extensie with co-contraction"] = {
        "CCI_LD_CO": float(CCI_LD_mean),
        "CCI_TM_CO": float(CCI_TM_mean)}

#%% CCI OPSLAAN IN JSON FILE ------------------------------------------------------------------------------
with open(save_path, 'w') as f:
    json.dump(cci_values, f, indent=4)

print(f"✅ CCI values saved to {filename}")