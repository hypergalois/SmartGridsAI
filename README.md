# SGBA2 – Smart Grids Based Architecture 2

**Modular MLOps pipeline for time-series forecasting and smart energy decisions (consumption, PV, pricing).**

## 📌 Overview

SGBA2 is a modular, scalable system for forecasting energy consumption, photovoltaic (PV) production, and electricity pricing at the household level. Built on a full MLOps stack using MLRun and MinIO, the system enables data-driven energy decisions through versioned models and reproducible pipelines.

## ⚙️ Components

### 🔋 Energy Forecasting Models

- **Consumption**: Hourly prediction per household using `XGBoostRegressor`.
- **Production (PV)**: Forecast per household using `Prophet` with holiday and weather regressors.
- **Electricity Pricing**: National-level prediction based on temperature, demand, and generation data.

### 🧠 Intelligent Architecture

- **Modular pipelines** via MLRun.
- **Model versioning, metrics logging, artifact storage** (e.g. `.pkl`, `.png`, `.csv`).
- **MinIO** for dataset and model persistence.
- **Future integration**: Reinforcement Learning agent for energy trade decisions
