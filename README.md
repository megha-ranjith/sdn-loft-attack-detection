
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

