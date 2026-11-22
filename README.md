# Federated Learning for Healthcare Data Security

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/notebooks/federated_learning_healthcare_security.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Learn how hospitals can train an AI model together without sharing a single patient record.**

This is a beginner-friendly, hands-on research notebook. It runs in Google Colab with one click. No installation is needed.

> **Disclaimer:** This is an educational simulation on a public dataset. It is **not** a medical device and must not be used for diagnosis or treatment.

---

## Table of Contents

1. [The Big Idea (in plain words)](#the-big-idea-in-plain-words)
2. [What You Will Learn](#what-you-will-learn)
3. [How It Works](#how-it-works)
4. [Quick Start](#quick-start)
5. [What Is Inside the Notebook](#what-is-inside-the-notebook)
6. [Results](#results)
7. [How to Read the Results](#how-to-read-the-results)
8. [Key Terms Explained](#key-terms-explained)
9. [Try It Yourself (Experiments)](#try-it-yourself-experiments)
10. [Limitations](#limitations)
11. [Future Work](#future-work)
12. [FAQ](#faq)
13. [References](#references)
14. [Author](#author)

---

## The Big Idea (in plain words)

Imagine 5 hospitals that all want a smarter heart disease prediction model.

- **The problem:** Each hospital has only a few patients, so its own model is weak. Combining all the data would help, but privacy laws (HIPAA, GDPR) and trust issues stop hospitals from sharing patient records.
- **The solution, Federated Learning (FL):** The data stays in each hospital. Only the *lessons the model learned* travel to a central server, where they are averaged into one shared model.

**Analogy:** Five students study from their own private notebooks. Instead of swapping notebooks, they each send a short summary of what they learned to a teacher. The teacher combines the summaries and sends back an improved version. Nobody ever sees another student's notebook.

**But there is a catch:** Even the summaries (model updates) can leak private information. So this project also tests three protections: **differential privacy**, **secure aggregation**, and a **privacy attack** to measure how well they work.

---

## What You Will Learn

- What federated learning is and why healthcare needs it
- How to simulate hospitals with *different* patient populations (non-IID data)
- How the **FedAvg** algorithm works, written from scratch in NumPy
- How **differential privacy** trades accuracy for privacy
- How **secure aggregation** hides each hospital's update from the server
- How to **audit privacy** with a membership inference attack
- How to run experiments over several random seeds and report results honestly

**Level:** Beginner to intermediate. You should know basic Python. No deep learning knowledge is needed.

---

## How It Works

```
   Hospital 1        Hospital 2        Hospital 3     ...   Hospital 5
 [patient data]    [patient data]    [patient data]         [patient data]
   (stays here)      (stays here)      (stays here)           (stays here)
        |                 |                 |                      |
   train locally     train locally     train locally          train locally
        |                 |                 |                      |
        v                 v                 v                      v
   clip + (mask)     clip + (mask)     clip + (mask)          clip + (mask)
        \                 |                 |                     /
         \________________|_________________|____________________/
                                   |
                                   v
                      Central server averages the updates
                          (+ adds privacy noise)
                                   |
                                   v
                         New improved global model
                                   |
              sent back to all hospitals --> repeat for many rounds
```

**One round, step by step:**

1. The server sends the current global model to every hospital.
2. Each hospital trains it for a few epochs on its **own** data.
3. Each hospital sends back only the **change** in the model weights.
4. *(Optional privacy)* Each update is **clipped** to a maximum size, and **masks** hide it from the server.
5. The server averages the updates and *(optional)* adds **noise**.
6. Repeat for 30 rounds.

---

## Quick Start

### Option 1: Google Colab (easiest, no setup)

1. Click the **Open In Colab** badge at the top of this page.
2. Click **Runtime → Run all**.
3. Wait 1-2 minutes. Scroll down to see the tables and charts.

### Option 2: Run on your own computer

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install numpy pandas scikit-learn matplotlib ucimlrepo jupyter
jupyter notebook notebooks/federated_learning_healthcare_security.ipynb
```

---

## What Is Inside the Notebook

| Part | What happens | What you learn |
|---|---|---|
| **1. Data and hospitals** | Splits the dataset across 5 hospitals with different patient mixes | Why real hospitals have non-IID data |
| **2. Federated training** | Builds logistic regression and FedAvg from scratch | How model averaging works |
| **3. Differential privacy** | Clips updates and adds Gaussian noise | The privacy-utility trade-off |
| **4. Secure aggregation** | Masks updates so the server sees only the total | How to hide individual updates |
| **5. Membership inference** | Attacks the model to test if it leaks training membership | How to measure privacy |
| **6. Results** | Averages 5 random seeds, makes plots and a CSV | How to report results honestly |

---

## Results

Run the notebook, then copy your numbers from `results_fl_healthcare.csv` into this table.
Test AUC measures accuracy (higher is better). Attack AUC measures privacy leakage (closer to 0.5 is better).

| Model | Test AUC (mean ± std) | Attack AUC (mean ± std) |
|---|---|---|
| Local-only (average of hospitals) | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Centralized (pooled data) | 0.00 ± 0.00 | 0.00 ± 0.00 |
| FedAvg | 0.00 ± 0.00 | 0.00 ± 0.00 |
| DP-FedAvg (noise = 0.5) | 0.00 ± 0.00 | 0.00 ± 0.00 |
| DP-FedAvg (noise = 1.0) | 0.00 ± 0.00 | 0.00 ± 0.00 |
| DP-FedAvg (noise = 2.0) | 0.00 ± 0.00 | 0.00 ± 0.00 |
| DP-FedAvg (noise = 4.0) | 0.00 ± 0.00 | 0.00 ± 0.00 |

*(Replace the zeros with your real values.)*

Add your charts here after running the notebook:

```markdown
![Convergence](images/convergence.png)
![Privacy vs utility](images/privacy_utility.png)
```

---

## How to Read the Results

- **Local-only** is each hospital training alone. It is often the weakest because each hospital sees few patients.
- **Centralized** pools all data. It is the accuracy ceiling, but it breaks privacy rules, so it is only a comparison.
- **FedAvg** should come close to centralized accuracy **without moving any patient data**. This is the main promise of FL.
- **DP-FedAvg** adds noise. As noise goes up, accuracy usually goes down while privacy improves. This is the **privacy-utility trade-off**.
- **Attack AUC:** 0.5 means the attacker is guessing (good). A value well above 0.5 means the model leaks who was in the training data.

**Be careful:** The dataset is small, so always look at the standard deviation across seeds before drawing conclusions. Privacy attacks on tiny models are weak, so differences may be small. This notebook is a **method demonstration**, not proof that one method is better.

---

## Key Terms Explained

| Term | Simple meaning |
|---|---|
| **Federated learning** | Training one model across many sites without moving the data |
| **FedAvg** | The server averages the model updates sent by each site |
| **Non-IID data** | Each site has a different kind of patients (the realistic case) |
| **Model update** | The change in model weights after local training |
| **Differential privacy (DP)** | A mathematical way to limit what a model can reveal about any one person |
| **Clipping** | Capping how large one site's update can be |
| **Noise multiplier** | How much random noise is added. More noise means more privacy and less accuracy |
| **Secure aggregation** | Masks that cancel out, so the server only learns the sum |
| **Membership inference** | An attack that guesses whether a record was in the training set |
| **AUC** | A score from 0.5 (guessing) to 1.0 (perfect) for how well a model separates sick from healthy |

---

## Try It Yourself (Experiments)

Change one thing at a time and watch what happens:

| Experiment | What to change | What to expect |
|---|---|---|
| More different hospitals | `alpha` in `prepare()`: try 0.1, 0.5, 5.0 | Smaller alpha means harder, more unequal data |
| Stronger privacy | `noise_levels` list: add 8.0 | Accuracy drops further |
| More training | `rounds` in `federated_train()`: try 10 vs 60 | See how fast FL converges |
| More hospitals | `K` in `prepare()`: try 3 vs 10 | Less data per hospital |
| Different clip size | `clip`: try 0.1, 1.0, 5.0 | Too small slows learning, too large weakens privacy |

---

## Limitations

- Hospitals are **simulated** from one public dataset, not real multi-site data.
- The dataset is small (about 300 patients), so results vary between runs.
- The notebook reports a **noise level**, not a formal privacy guarantee (epsilon, delta).
- The DP noise assumes a trusted server or secure aggregation placed in front of it.
- Secure aggregation is a **simplified demo**. Real protocols also handle key exchange and hospitals dropping out.
- Feature scaling uses pooled statistics. Real FL needs federated preprocessing.
- The model is a simple logistic regression, not a deep network.

---

## Future Work

1. Add formal privacy accounting with **Opacus** or **TensorFlow Privacy**.
2. Handle uneven data better with **FedProx**, **SCAFFOLD**, or personalized FL.
3. Test stronger attacks such as gradient inversion and shadow-model membership inference.
4. Combine **secure aggregation + differential privacy** with dropout handling.
5. Use real healthcare data and standards (**MIMIC-IV**, FHIR, chest X-rays).
6. Study **fairness** across hospitals and patient groups.
7. Explore **homomorphic encryption** and trusted hardware.

---

## FAQ

**Is my data uploaded anywhere?**
No. The notebook downloads a public dataset and runs inside your own Colab session.

**Why does the notebook say "fallback dataset"?**
If the UCI download fails, the notebook uses a built-in scikit-learn dataset so it still runs. Results will differ slightly.

**Why is the accuracy so high for every method?**
The dataset is small and easy. The goal here is to compare privacy methods, not to reach record accuracy.

**Can I use this for a real hospital?**
No. It is an educational simulation. Real deployments need legal review, security audits, and clinical validation.

**Can I use a different dataset?**
Yes. Edit the `load_data()` function so it returns a feature table `X` and labels `y` (0 or 1).

---

## References

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* (FedAvg)
- Dwork & Roth (2014). *The Algorithmic Foundations of Differential Privacy.*
- Abadi et al. (2016). *Deep Learning with Differential Privacy.*
- Bonawitz et al. (2017). *Practical Secure Aggregation for Privacy-Preserving Machine Learning.*
- Shokri et al. (2017). *Membership Inference Attacks Against Machine Learning Models.*
- Rieke et al. (2020). *The Future of Digital Health with Federated Learning.*
- Dataset: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland subset)

---

## Author

**Amit Thakur**: MSc IT (AI & ML) | Healthcare IT | AWS Certified
[LinkedIn]([https://www.linkedin.com/in/amit-thakur-b72517254]) | [GitHub]([https://github.com/mitthakur12])

If this project helped you, please give it a star ⭐
