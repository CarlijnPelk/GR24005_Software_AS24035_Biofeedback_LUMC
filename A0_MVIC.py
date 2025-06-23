#%% IMPORTEREN PACKAGES --------------------------------------------------------------------------------------
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

#%% POP-UP-----------------------------------------------------------------------------------------------------
root = tk.Tk()

# participant nummer kiezen
ptnr   = simpledialog.askstring( "Participant Info",   "Participant number:", parent=root )
root.withdraw()

# mvic meting type kiezen
v = tk.IntVar(value=1)
MVIC_options = {"abduction":1, "adduction":2}
reverse_mvic = {val:key for key,val in MVIC_options.items()}

frame = tk.Toplevel(root)
tk.Label(frame, text="Choose AB- or ADduction:").pack(padx=20, pady=10)
for text, val in MVIC_options.items():
    tk.Radiobutton(frame, text=text, variable=v, value=val).pack(anchor="w", padx=20)

def submit_choice():
    global mvic_type
    mvic_type = reverse_mvic[v.get()]
    print("You chose MVIC in", mvic_type)
    frame.destroy()
    root.quit()

tk.Button(frame, text="Submit", command=submit_choice).pack(pady=10)

root.mainloop()   
root.destroy()
print("✅ Questions completed. Continuing…")

#%% LAAD MVIC FILE OF MAAK NIEUWE------------------------------------------------------------------------------
filename = f"mvic_{ptnr}.json"
folder_name = "mvic"

folder_path = os.path.join(os.getcwd(), folder_name) 
os.makedirs(folder_path, exist_ok=True)

save_path = os.path.join(folder_path, filename)

if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        mvic_values = json.load(f)
else:
    mvic_values = {"participant": ptnr}

#%% PRE-PROCESSING FUNCTIES -------------------------------------------------------------------------------
#Lowpass-filter functie 
def lowpass(data, cutoff, fs, poles=5):
    sos = scipy.signal.butter(N=poles, Wn=cutoff, btype='low', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)

#Root Mean Square functie
def moving_rms(data, window_ms, fs):
    w = int(window_ms/1000 * fs) or 1
    sq = data**2
    kernel = np.ones(w) / w
    rms = np.empty_like(data)
    for ch in range(data.shape[1]):
        ms = np.convolve(sq[:, ch], kernel, mode='same')
        rms[:, ch] = np.sqrt(ms)
    return rms

#Highpass-filter functie
def highpass(data, cutoff, fs, poles=5):
    sos = scipy.signal.butter(N=poles,Wn=cutoff,btype='high', fs=fs,output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)

#%% DATA ACQUISITIE ----------------------------------------------------------------------------------------
device = "Dev2"
channels = ["4", "6", "7"]
sample_rate = 1000  # Hz
n_channels = len(channels)
data_list = []

# Start en stopscherm maken 
window = tk.Tk()
window.title("Start MVIC meting")

label = tk.Label(window, text="Druk op de knop om MVIC-meting te starten (5 seconden):", font=("Arial", 12))
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

#%% STARTFUNCTIE --------------------------------------------------------------------------------------------
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

#%% SIGNALEN PREPROCESSEN -----------------------------------------------------------------------------------
emg_data = np.array(data_list) 
emg_hp = highpass(emg_data, cutoff=20, fs=1000)   
emg_rms = moving_rms(emg_hp, window_ms=50, fs=1000) 
emg_filtered = lowpass(emg_rms, cutoff=450, fs=1000)

#%% MVIC OPSLAAN PER METING --------------------------------------------------------------------------------
mvic_DM_save = []
if mvic_type == "abduction":
    dm_channel = emg_filtered[:, 0]
    mvic_DM = np.mean(np.sort(dm_channel)[-100:])
    print("\n📈 MVIC value deltoideus medialis: ", mvic_DM)
    mvic_values["abduction"] = {
        "mvic_DM": float(mvic_DM)}
    
elif mvic_type == "adduction":
    ld_channel = emg_filtered[:, 1]
    tm_channel = emg_filtered[:, 2]
    mvic_LD = np.mean(np.sort(ld_channel)[-100:])
    mvic_TM = np.mean(np.sort(tm_channel)[-100:])
    print("\n📈 MVIC values \n latissimus dors: ", mvic_LD, "\n teres major: ", mvic_TM)
    mvic_values["adduction"] = {
        "mvic_LD": float(mvic_LD),
        "mvic_TM": float(mvic_TM)}

#%% MVIC OPSLAAN IN JSON FILE ------------------------------------------------------------------------------------------
with open(save_path, 'w') as f:
    json.dump(mvic_values, f, indent=4)

print(f"✅ MVIC values saved to {filename}")