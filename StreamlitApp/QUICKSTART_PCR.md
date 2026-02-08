# Quick Start Guide: PCR Application

## 🚀 Running the Streamlit App

### Option 1: Quick Launch
```bash
cd d:\Codes\TTK4260-2026\StreamlitApp
streamlit run app.py
```

Then click on **"07 PCR Application"** in the sidebar.

### Option 2: Direct Launch
```bash
cd d:\Codes\TTK4260-2026\StreamlitApp
streamlit run pages/07_PCR_Application.py
```

## 📊 What You'll See

The app opens in your browser with **8 tabs**:

1. **📊 Data Overview** - Explore the 26,304 Norwegian energy samples
2. **1️⃣ MLR Baseline** - See multicollinearity issues with bootstrap
3. **2️⃣ PCA Structure** - Scree plot and variance explained
4. **3️⃣ PCR Performance** - Try different k values, see the asymmetry
5. **4️⃣ PCA Diagnostics** - Interactive score plots (try PC5 vs PC6!)
6. **5️⃣ Feature Analysis** - WindSpeed vs radiation variables
7. **6️⃣ PC Interpretation** - What does each PC capture?
8. **7️⃣ PC-Response Correlation** - The key mismatch revealed
9. **8️⃣ R² vs k** - Watch PCR improve as you add components

## 💡 Key Things to Try

### In Tab 3 (PCR Performance):
- **Slide k from 1 to 14** → Watch WindPower R² jump at k=6!

### In Tab 4 (PCA Diagnostics):
- **Set X-axis to PC5, Y-axis to PC6** → See the wind gradient that PC1-PC2 miss!

### In Tab 7 (PC-Response Correlation):
- **Look for yellow highlights** → Find which PCs matter for each response

### In Tab 8 (R² vs k):
- **Hover over the lines** → See exact R² values at each k

## 🎯 The Main Lesson

**Same data, same PCA, same k=3:**
- PVPower: R² ≈ 0.85 ✅ Works great!
- WindPower: R² ≈ 0.13 ❌ Fails badly!

**Why?** PCA ranks by X-variance, not predictive power. The wind signal lives in PC5-PC6 (low variance), while PV signal lives in PC1 (high variance).

## 📱 Comparing Notebook vs App

| Feature | Notebook | Streamlit App |
|---------|----------|---------------|
| Navigation | Sequential cells | 8 organized tabs |
| Score plot PCs | Fixed PC1-PC2 | **Interactive dropdowns** |
| PCR k value | Fixed k=3 | **Interactive slider** |
| Interactivity | Static once run | **Live updates** |
| Best for | Deep reading | Quick exploration |

## 🔧 Troubleshooting

### Data not loading?
Check that these files exist:
```
StreamlitApp/data/X.txt
StreamlitApp/data/Y.txt
```

### Plots not showing?
Make sure you have the required packages:
```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### App is slow?
The bootstrap (Tab 1) and R² calculation (Tab 8) take a few seconds. This is normal!

## 📚 Learn More

- **Full notebook**: `PrincipalComponentRegression/Notebook/PCR_Application.ipynb`
- **Implementation details**: `StreamlitApp/PCR_APPLICATION_IMPLEMENTATION.md`
- **Course topics**: `StreamlitApp/README.md`

---

**Enjoy exploring!** 🎉
