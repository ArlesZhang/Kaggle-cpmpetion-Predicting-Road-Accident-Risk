## Predicting Road Accident Risk — Intelligent ML System Optimization Pipeline

### **Project Vision**

This repository is the **first milestone** in my 530-day research and system engineering roadmap — a practical experiment in **AI compiler optimization + intelligent ML pipeline design**.
Built around the **Kaggle "Predicting Road Accident Risk"** competition, it serves as a **sandbox for system-level innovation**, bridging research, productization, and global competition readiness.

---

### **Core Objectives**

| Goal                          | Description                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Data Engineering**       | Automated data cleansing, transformation, and synthesis for geospatial-temporal datasets.             |
| **Model Optimization**     | Multi-model experimentation with LightGBM, CatBoost, XGBoost, and ensemble fusion (meta-blending).    |
| **Pipeline Abstraction**   | Modularized ML pipeline supporting automated validation, feature iteration, and deployment readiness. |
| **System Deployability**   | Fully reproducible Docker environment; ready for integration with FastAPI or MLflow.                  |
| **AI Training Efficiency** | Explore compiler-level acceleration (MLIR) and scheduling optimization for training workloads.        |

---

### **Strategic Role in My Research Plan**

This Kaggle project forms **Project-1** in my broader AI systems trilogy:

| Project       | Theme                       | Research Direction                                                                           |
| ------------- | --------------------------- | -------------------------------------------------------------------------------------------- |
| **Project 1** | Data → Model → Ensemble     | Kaggle competition system, validation of pipeline performance and feature optimization       |
| **Project 2** | MLIR + NPU Scheduling       | Build compiler-level optimization pass for ML inference/training workloads                   |
| **Project 3** | Distributed Training System | Design a modular scheduling layer integrating pipeline orchestration + AI compiler co-design |

Together, these projects support my long-term goal:

> Becoming a **global expert in AI compiler optimization and distributed ML infrastructure**, capable of merging **open-source innovation with industrial-scale system design**.

---

### 🏆 **Competition & Publication Targets**

This repository and its successors will evolve toward **real competition and research participation**, including:

* **Kaggle International Competitions** — from Playground series to **NeurIPS / ARC Prize 2025** level challenges.
* **2025 琶洲AI创新大赛** — prototype-level system deployment showcasing intelligent ML pipeline design.
* **Workshop Submissions @ MLSys / NeurIPS / ICLR** — derived from this system’s architectural innovations.

---

### **Current Progress**

| Category            | Status                                                                        |
| ------------------- | ----------------------------------------------------------------------------- |
| Public Leaderboard  | 🥉 **0.05554 RMSE (Top 18%)** — Fine Ensemble v1                              |
| System Architecture | Modular pipeline with automated ensembling and validation                     |
| Experiment Tracking | Manual logging + Kaggle versioning (to be extended to MLflow)                 |
| Research Notes      | Drafting `ml_pipeline_compiler_ideas.md` (MLIR + runtime scheduling concepts) |

---

### **Technical Highlights**

* **Multi-model Ensemble System:** blending LightGBM, CatBoost, and XGBoost with tunable weights.
* **Feature-driven Improvement:** temporal aggregation, location encoding, and density-based risk modeling.
* **Reproducible ML Environment:** Python + Docker + Kaggle CLI workflow automation.
* **System Optimization Hooks:** ready for future integration with MLIR optimization and ONNX quantization passes.

---

### **Future Roadmap (2025 Q4 → 2026 Q2)**

| Quarter     | Milestone                                                               |
| ----------- | ----------------------------------------------------------------------- |
| **Q4 2025** | Publish enhanced ensemble + Dockerized API demo (`FastAPI + Streamlit`) |
| **Q1 2026** | Integrate MLIR optimization pass for model inference pipeline           |
| **Q2 2026** | Extend to distributed training scheduler (Ray + custom MLIR backend)    |
| **Q3 2026** | Submit to **NeurIPS/MLSys workshop**, apply to **ARC Prize 2026**       |

---

### **Repository Layout**

```
road_accident/
├── data/                     # Training & test data
├── models/                   # Model scripts and saved weights
├── scripts/                  # Preprocessing, ensemble, analysis
├── outputs/                  # Submissions, logs, reports
├── docker/                   # Dockerfile & environment setup
└── README.md                 # Documentation (this file)
```

---

### **Vision Statement**

> This repository marks the starting point of a **system-level AI engineering journey** —
> evolving from Kaggle competitions to **real-world compiler-driven ML optimization systems**,
> and ultimately, toward **NeurIPS-level AI infrastructure research and commercialization**.

