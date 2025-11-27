
# SDN LOFT Attack Detection

This project is a **simple simulation-based prototype** for detecting and mitigating Low-Rate Flow Table Overflow (LOFT) attacks in Software-Defined Networks (SDN). It demonstrates how a controller can use **machine learning (Random Forest)** to distinguish between normal and malicious flows and automatically evict attack entries from the flow table.

---

## 🚀 Overview

- Simulates an SDN switch flow table with limited capacity.
- Generates both **normal** and **attack** flows (LOFT-style low-rate, short-duration flows).
- Extracts basic flow features (packet count, bytes, duration, ports, protocol).
- Trains a **Random Forest classifier** to label flows as `normal` or `attack`. 
- Periodically scans the flow table, detects malicious entries, and **removes them in real time**.
- Prints a final detection report with accuracy and flow table occupancy.

This repository is intended for **educational and micro-project use** (M.Tech – Advanced Computer Networks).

---

## ⚙️ Technology Stack

- **Language:** Python 3.10+
- **Libraries:**
  - `numpy`
  - `pandas` (optional, used in dataset prep)
  - `scikit-learn`
- **Environment:** Tested on VS Code + terminal (Windows / Linux)

---

## 📂 Project Structure

```
sdn-loft-attack-detection/
├── sdn_attack_detector.py # Main simulation + ML-based detection
├── README.md # Project documentation
└── (optional) report/ # LaTeX/PDF micro-project report and figures
```

---

## 🔧 Installation & Setup

1. **Clone the repository**

```
git clone https://github.com/megha-ranjith/sdn-loft-attack-detection.git
cd sdn-loft-attack-detection
```

2. **Create and activate a virtual environment (optional but recommended)**

```
python -m venv venv
```
Windows
```
venv\Scripts\activate
```

Linux / macOS
```
source venv/bin/activate
```

3. **Install dependencies**

```
pip install numpy pandas scikit-learn
```

---

## ▶️ Running the Simulation

```
python sdn_attack_detector.py
```

You will see:

1. **Model training phase**  
   - Dataset generation (normal vs attack flows)  
   - Random Forest training and printed metrics (Accuracy, Precision, Recall, F1-score)

2. **Traffic simulation phase**  
   - Mixed normal + attack flows added to the flow table  
   - Periodic detection:  
     - `[ALERT] Detected X malicious flows!`  
     - `[ACTION] Removing malicious flows...`  
   - Status lines showing:
     - Flow table occupancy
     - Total flows processed
     - Normal vs attack flows
     - Detected and blocked attacks

3. **Final report**  
   - Total flows, number of attacks, detection rate, and final occupancy.

---

## 📊 What This Demonstrates

- How limited flow table capacity in SDN switches can be attacked using **low-rate overflow**.
- How **ML-based detection** (Random Forest) can detect such attacks using simple flow features.
- Real-time **mitigation** by evicting malicious flow rules, preserving resources for legitimate traffic.

This is a **simplified version** of ideas inspired by recent research on LOFT attack detection, adapted for a small, runnable micro-project.

---

## 🧪 Customization

You can tweak basic parameters directly in `sdn_attack_detector.py`:

- Flow table capacity:
  - `SDNFlowTableSimulator(capacity=100)`
- Simulation duration:
  - `simulate_traffic(duration=30, attack_probability=0.3)`
- Attack probability:
  - `attack_probability` argument in `simulate_traffic`

Experimenting with these values is a good way to explore how the system behaves under different attack intensities.

---

## 📖 Academic Context

This code was developed as part of an **M.Tech micro-project** for the course *Advanced Computer Networks*, focusing on SDN security and anomaly detection using machine learning.

---

## ✍️ Author

- **Megha Ranjith**  
- GitHub: [@megha-ranjith](https://github.com/megha-ranjith)

---

## 🔒 Disclaimer

This is a **simulation-only, educational prototype**.  
It is not intended for direct deployment in production SDN environments.
