# BRFSS Health Insights Dashboard (Python • Plotly Dash)

An interactive health analytics dashboard built on **CDC BRFSS prevalence data**.  
The application enables users to explore health trends over time, compare geographic regions, and analyze demographic disparities with statistical confidence intervals — optimized for large datasets using **Parquet caching**.

---

##  What This Dashboard Does

- **Hierarchical exploration**: Health Class → Topic → Question
- **Temporal analysis**: Trends from **2011–2023**
- **Demographic breakouts**:
  - Gender
  - Age Group
  - Race / Ethnicity
  - Education
  - Income
- **Geographic comparison**:
  - States (50+)
  - Census Regions (4)
  - Census Divisions (9)
  - Interactive U.S. choropleth map
- **Insights & Data view**:
  - Sample size
  - National prevalence
  - Trend deltas
  - Geographic range
  - 95% Confidence Interval tables
- **Performance-aware design**:
  - Loads a **>1GB dataset**
  - Automatically caches cleaned data using **Parquet (PyArrow)** for faster reloads

---

## 📸 Screenshots

Place screenshots in `assets/screenshots/` and ensure filenames match below.

| Overview | Demographics |
|--------|--------------|
| ![](assets/screenshots/overview_tab.png) | ![](assets/screenshots/detailed_%20demographics_tab.png) |

| Geography | Compare | Insights |
|---------|---------|----------|
| ![](assets/screenshots/geography_tab.png) | ![](assets/screenshots/compare_tab.png) | ![](assets/screenshots/insights_tab.png) |

---

## 🧰 Tech Stack

- **Python**
- **Plotly Dash**
- Dash Bootstrap Components
- **Pandas**, NumPy
- **PyArrow / Parquet** (data caching & performance)

---

## 🚀 Getting Started (Local Setup)

### 1️⃣ Create environment & install dependencies
```bash
conda create -n brfssdash python=3.10 -y
conda activate brfssdash
pip install -r requirements.txt

## 2️⃣ Add the Dataset (Not Included in Repository)

This repository does **not** include the BRFSS dataset due to its size (**>1GB**).

Create the following directory in the project root:

data/

Place the BRFSS CSV file here:

data/Prevalence_Data.csv

The dataset is sourced from the **CDC BRFSS Prevalence Data Portal**.

---

## 3️⃣ Run the Dashboard

Start the Dash application using:

```bash
python app.py
Open the dashboard in your browser:
http://127.0.0.1:8050/

## 4️⃣ Data Loading & Performance Optimization
The application uses a smart data-loading pipeline to efficiently handle large datasets:

Attempts to load from a Parquet cache first (fast)

Falls back to loading the CSV file if the cache is missing (slower, one-time)

Applies preprocessing steps:

ResponseID and BreakoutID normalization

Removal of aggregate rows (US, UW)

Year type enforcement and validation

Automatically saves a Parquet cache after the first successful load:

data/brfss_prevalence.parquet
Subsequent runs load directly from Parquet, resulting in significantly faster startup times.

## 5️⃣ Key Engineering Decisions
Parquet caching to support repeated analysis on large datasets

Semantic color encoding for response categories (Yes / No)

Modular aggregation utilities to support multiple analytical views

Statistical context via 95% confidence intervals

Graceful filtering across time, geography, and demographic dimensions

## 6️⃣ Project Context & Collaboration
Developed as part of a Northeastern University data visualization course.

We explored parallel implementations using different technology stacks (Python Dash vs R Shiny).
This repository contains my end-to-end Python Dash implementation, focusing on data preprocessing, performance optimization, and interactive analytics design.

## 7️⃣ Roadmap (Future Improvements)
Add a lightweight demo mode with sampled data

Improve error handling when the dataset is missing

Add automated tests for preprocessing utilities

Optional cloud deployment for live demo access

## 8️⃣ License
This project is intended for educational and portfolio purposes.