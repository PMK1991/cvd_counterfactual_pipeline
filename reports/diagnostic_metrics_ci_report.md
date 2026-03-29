# Confidence Intervals for Downstream Diagnostic Metrics
## CVD Counterfactual Analysis - 95% Confidence Intervals

**Analysis Details:**
- Bootstrap iterations: 100
- Confidence level: 95%
- Total successful counterfactuals: 94
- Method: Percentile-based bootstrap CI

---

## Summary of Results with 95% Confidence Intervals

### 📊 **Total Successful Counterfactuals**
- **Count:** 94 (fixed across all bootstrap samples)
- **95% CI:** [94, 94]

---

### 💓 **Chest Pain Type (cp)**
**Proportion Changed:**
- **Mean:** 84.1%
- **95% CI:** [77.1%, 92.6%]
- **Interpretation:** Between 77-93% of successful counterfactuals show changes in chest pain type

**Count Changed:**
- **Mean:** 79.0
- **95% CI:** [72.5, 87.0]

---

### 🏃 **Exercise-Induced Angina (exang)**

**Resolved Angina (1 → 0):**
- **Mean:** 62.7%
- **95% CI:** [55.3%, 71.3%]
- **Count:** 59.0 (95% CI: [52.0, 67.0])
- **Interpretation:** Approximately 56-71% of successful CFs show resolution of exercise-induced angina

**Developed Angina (0 → 1):**
- **Mean:** 4.4%
- **95% CI:** [1.1%, 9.1%]
- **Count:** 4.1 (95% CI: [1.0, 8.5])
- **Interpretation:** Only 1-9% of successful CFs develop new exercise-induced angina

---

### 📉 **ST Depression (oldpeak)**

**Proportion with Reduced oldpeak:**
- **Mean:** 72.6%
- **95% CI:** [62.2%, 80.9%]
- **Count:** 68.3 (95% CI: [58.5, 76.0])

**Original oldpeak:**
- **Mean:** 1.75 mm
- **95% CI:** [1.49, 2.06] mm
- **Median:** 1.56 mm (95% CI: [1.40, 2.00])

**Counterfactual oldpeak:**
- **Mean:** 0.28 mm
- **95% CI:** [0.18, 0.39] mm
- **Median:** 0.00 mm (95% CI: [0.00, 0.00])

**Mean Change:**
- **Mean:** -1.47 mm
- **95% CI:** [-1.79, -1.15] mm
- **Interpretation:** ST depression improves by 1.15-1.79 mm on average (negative = improvement)

---

### ❤️ **Maximum Heart Rate (thalach)**

**Proportion with Increased thalach:**
- **Mean:** 70.6%
- **95% CI:** [63.3%, 78.7%]
- **Count:** 66.4 (95% CI: [59.5, 74.0])

**Original thalach:**
- **Mean:** 127.6 bpm
- **95% CI:** [124.7, 131.2] bpm
- **Median:** 125.0 bpm (95% CI: [119.0, 130.0])

**Counterfactual thalach:**
- **Mean:** 145.8 bpm
- **95% CI:** [142.6, 150.4] bpm
- **Median:** 141.7 bpm (95% CI: [137.0, 150.0])

**Mean Change:**
- **Mean:** +18.2 bpm
- **95% CI:** [+13.4, +23.4] bpm
- **Interpretation:** Maximum heart rate increases by 13-23 bpm on average (positive = improvement)

---

### 📈 **ST Slope (slope)**

**Proportion with Decreased slope (improved):**
- **Mean:** 80.2%
- **95% CI:** [71.3%, 87.2%]
- **Count:** 75.4 (95% CI: [67.0, 82.0])
- **Interpretation:** 71-87% of successful CFs show improvement in ST slope

---

### 🔬 **Resting ECG (restecg)**

**No Change:**
- **Mean:** 43.0%
- **95% CI:** [35.1%, 52.2%]
- **Count:** 40.4 (95% CI: [33.0, 49.1])

**Improved (decreased):**
- **Mean:** 32.4%
- **95% CI:** [20.7%, 41.0%]
- **Count:** 30.5 (95% CI: [19.5, 38.6])

**Worsened (increased):**
- **Mean:** 24.6%
- **95% CI:** [17.0%, 33.6%]
- **Count:** 23.1 (95% CI: [16.0, 31.6])

---

## 🎯 Key Insights

### Most Consistent Changes (Narrow CIs):
1. **Total successful CFs:** No variability (fixed at 94)
2. **Median CF oldpeak:** Always 0.00 mm (100% consistency)
3. **Mean thalach change:** Relatively narrow CI (±5 bpm around mean)

### Most Variable Changes (Wide CIs):
1. **Resting ECG changes:** Wide CIs across all categories (±8-10%)
2. **Exang 0→1:** Wide relative CI (1-9%), though absolute numbers are small
3. **Oldpeak reduction:** Moderate variability (±9%)

### Strongest Effects (Largest Improvements):
1. **ST Depression (oldpeak):** -1.47 mm average reduction (CI: -1.79 to -1.15)
2. **Maximum Heart Rate:** +18.2 bpm average increase (CI: +13.4 to +23.4)
3. **ST Slope:** 80.2% improved (CI: 71.3% to 87.2%)

### Most Reliable Improvements:
1. **Chest pain type changes:** 84.1% (CI: 77.1% to 92.6%)
2. **ST slope improvement:** 80.2% (CI: 71.3% to 87.2%)
3. **ST depression reduction:** 72.6% (CI: 62.2% to 80.9%)

---

## 📝 Statistical Notes

- **Bootstrap Method:** Percentile-based confidence intervals from 100 bootstrap resamples
- **Sampling:** With replacement from 94 successful counterfactuals
- **Confidence Level:** 95% (α = 0.05)
- **CI Width:** Indicates precision of estimates (narrower = more precise)

---

## 🔍 Interpretation Guide

**For Percentages:**
- The true population percentage likely falls within the CI range
- Narrower CIs indicate more stable/reliable estimates
- Example: "62.7% (CI: 55.3%-71.3%)" means we're 95% confident the true percentage is between 55.3% and 71.3%

**For Continuous Measures (oldpeak, thalach):**
- Negative changes in oldpeak = improvement (less ST depression)
- Positive changes in thalach = improvement (higher max heart rate)
- CIs that don't include zero indicate statistically significant changes

**For Counts:**
- Absolute number of counterfactuals showing each characteristic
- Useful for understanding sample size and effect magnitude

---

*Generated from bootstrap analysis with 100 iterations*
*Data source: worked/working_counterfactuals_with_distances.csv*
