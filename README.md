# GlucoAUC
A Python pipeline for analysing continuous glucose monitoring (CGM) data, uncovering glucose patterns beyond spikes and averages. Compatible with LibreView data.

**Beyond glucose spikes: A personal continuous glucose monitoring (CGM) analytics pipeline.**

GlucoAUC is a Python-based analytics pipeline that processes continuous glucose monitoring data exported from FreeStyle Libre to uncover glucose patterns beyond conventional summary metrics.

## Motivation

Most CGM applications provide useful metrics such as average glucose, estimated HbA1c, time in range, and glucose trends. However, these summaries do not always make it easy to quantify the **duration and cumulative exposure of individual glucose elevations**.

Two glucose excursions may reach similar peaks but differ substantially in how long glucose remains elevated.

I developed GlucoAUC to investigate these differences by quantifying glucose excursion characteristics, including incremental area under the curve (iAUC), duration, and recovery dynamics.

The goal is to transform raw CGM readings into interpretable insights that support personal exploration of glucose patterns and lifestyle responses.

## Features

### Glucose Summary Statistics

- Mean, median, standard deviation, and coefficient of variation (CV)
- Interquartile range (IQR)
- Time in range (3.9–10.0 mmol/L)
- Time in tight range (3.9–7.8 mmol/L)
- Time above and below specified glucose thresholds
- Exploratory mean amplitude of glycemic excursions (MAGE) calculation

### Automated Glucose Excursion Detection

Identifies glucose elevations relative to a rolling baseline and calculates:

- **Incremental AUC (iAUC):** Cumulative glucose exposure above baseline
- **Peak magnitude:** Maximum glucose elevation relative to baseline
- **Excursion duration:** Time spent within a detected elevation episode
- **Time to peak:** Duration from excursion onset to maximum glucose
- **Recovery time:** Estimated time required for glucose to decline towards baseline
- **Descent rate:** Rate of glucose decline following the peak

### Visualisation and Trend Analysis

- Continuous glucose traces with reference ranges
- Average glucose patterns by time of day
- Glucose distribution histograms and daily box plots
- Individual glucose excursion profiles
- Weekly trends in iAUC, recovery time, and excursion frequency

## Tech Stack

**Language:** Python

**Libraries:**
- Pandas — Data preprocessing and manipulation
- NumPy — Numerical calculations
- SciPy — Signal processing, peak detection, and numerical integration
- Matplotlib and Seaborn — Data visualisation

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/GlucoAUC.git
cd GlucoAUC
```

### 2. Install dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn
```

### 3. Export CGM data

Export your glucose data as a CSV file from LibreView.

### 4. Run the analysis

```bash
python cgm_analysis.py path/to/your_data.csv
```

The pipeline generates summary statistics, excursion-level measurements, and visualisations in a `cgm_outputs/` directory.

**Privacy:** Do not commit personal CGM exports or identifiable health data to a public repository.

## Roadmap

- [ ] Refine and validate glucose excursion detection
- [ ] Improve handling of missing measurements and incomplete recovery episodes
- [ ] Implement additional glycemic variability metrics
- [ ] Add statistical analysis of longitudinal glucose trends
- [ ] Incorporate meal annotations for food-specific glucose response analysis
- [ ] Develop interactive visualisations and dashboards

## Limitations

GlucoAUC is an exploratory personal analytics project and is not intended for clinical diagnosis or treatment decisions.

Glucose excursion metrics are influenced by factors including meal composition, physical activity, medication, and sensor measurement variability. CGM data alone cannot directly measure insulin sensitivity or insulin secretion.

The current excursion detection and MAGE algorithms are experimental and have not yet undergone formal validation.

## Project Status

**Work in progress** — Initial implementation of the CGM analytics pipeline. Further methodological improvements, validation, and additional features are planned.
