# GR24005_Software_AS24035_Biofeedback_LUMC

# SAPS sEMG-Biofeedback Prototype

This repository contains the custom Python software for a prototype rehabilitation aid developed to assist patients with **Subacromial Pain Syndrome (SAPS)**. The prototype continuously monitors **surface electromyography (sEMG)** signals to detect co-contraction of the *latissimus dorsi* (LD) and *teres major* (TM) during contraction of the *deltoid medialis* (DM) and provides **real-time auditory biofeedback** to encourage sufficient co-contraction. The goal is to form the basis for a SAPS rehabilitation aid.

---

## 🚀 Features

* Acquisition and processing of multi-channel sEMG data via NI-DAQ hardware.
* Custom filtering (high-pass, low-pass, RMS smoothing).
* Personalized threshold determination for DM and CCI (Co-Contraction Index) values.
* Flexible CCI thresholds allowing adaptation during rehabilitation stages.
* Real-time biofeedback with \~530 ms latency using audio signals.
* Verification and validation tools for prototype sensitivity and specificity.
* Data storage (CSV) for reproducibility and further analysis.
* Visualization of EMG signals, CCI metrics, and biofeedback activity.

---

## 🗂 Repository Structure

```
📂 /scripts
 ├── A1_Determine_MVIC.py            # Record MVIC values for normalization
 ├── A2_Determine_CCI.py             # Record and compute CCI values (baseline)
 ├── A3_RealTime_Biofeedback.py      # Run real-time biofeedback system
 ├── B1_Verification.py              # Compute sensitivity/specificity of detection
 ├── B2_Validation.py                # Compare CCI with/without biofeedback + stats
📂 /mvic                             # Saved MVIC data (per participant)
📂 /cci measurements                 # Saved CCI data (per participant)
📂 /EMG_Output_ptnrX                 # Raw EMG data per participant (CSV)
📄 verification_results.json         # Stored verification test results
📄 README.md                         # Project documentation
```

## 💻 How to use

1️⃣ **MVIC Calibration**
Run `A1_Determine_MVIC.py` to record participant-specific MVIC values for DM and LD + TM.

2️⃣ **CCI Baseline Measurement**
Run `A2_Determine_CCI.py` to measure baseline CCI values with/without co-contraction in three static positions.

3️⃣ **Real-time Biofeedback**
Run `A3_RealTime_Biofeedback.py` for live sEMG monitoring and auditory feedback during rehabilitation exercises.

4️⃣ **Verification/Validation**

* `B1_Verification.py`: Calculate sensitivity/specificity of co-contraction detection.
* `B2_Validation.py`: Evaluate co-contraction level differences between conditions (with vs without biofeedback).

---

## 📌 Notes

⚠ This code is research-grade and was developed for a proof-of-concept prototype. It is not intended for clinical or commercial use without further validation.

⚠ Hardware-specific code (NI DAQ) requires the appropriate setup and drivers.

---

## 📚 Citation

If you use or build upon this code, please cite:

van der Hart, S., Pelk, C., Swanenberg, B., & Vlug, S. (2025). Developing a real-time sEMG-biofeedback prototype to stimulate co-contraction of the shoulder adductors during abductor contraction in subacromial pain syndrome rehabilitation. TU Delft / LUMC.

