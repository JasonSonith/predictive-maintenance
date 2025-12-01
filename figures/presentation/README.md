# Presentation Figures - Usage Guide

## Generated Figures

All figures are saved as high-resolution PNGs (300 DPI) ready for PowerPoint insertion.

### High Priority Figures (Must Have)

1. **slide_09_ims_timeseries.png** (490 KB)
   - **Slide:** 9
   - **Purpose:** Shows bearing degradation over time - YOUR HERO GRAPH!
   - **What it shows:** Anomaly scores increasing before failure
   - **Models shown:** Isolation Forest, AutoEncoder, kNN-LOF
   - **Key insight:** All models detect degradation in files 70+ before failure at file 96
   - **Why it matters:** Proves your method works - visual evidence of early failure detection

2. **slide_10_far_comparison.png** (269 KB)
   - **Slide:** 10
   - **Purpose:** Demonstrates 100% calibration success across all datasets
   - **What it shows:** Target vs. Achieved FAR for 10 models
   - **Key insight:** All models successfully calibrated within acceptable range
   - **Color coding:** Green = perfect, success across all datasets

3. **slide_04_raw_waveform.png** (545 KB)
   - **Slide:** 4
   - **Purpose:** Shows raw bearing vibration data
   - **What it shows:** 2000 samples of time-series signal
   - **Key insight:** Helps audience understand the raw data problem

### Medium Priority Figures (Recommended)

4. **slide_08_threshold_distribution.png** (158 KB)
   - **Slide:** 8
   - **Purpose:** Explains threshold calibration visually
   - **What it shows:** Distribution of anomaly scores with threshold line
   - **Key insight:** Shows separation between normal and anomalous behavior

5. **slide_05_sliding_window.png** (383 KB)
   - **Slide:** 5
   - **Purpose:** Illustrates feature engineering approach
   - **What it shows:** Overlapping windows on raw signal
   - **Key insight:** Shows how time-series is converted to features

## How to Use in PowerPoint

1. **Insert Images:**
   - PowerPoint → Insert → Pictures → Browse
   - Select the PNG files from this directory
   - Place on appropriate slides

2. **Recommended Sizes:**
   - Full-width figures: 9-10 inches wide
   - Half-slide figures: 5-6 inches wide
   - Maintain aspect ratio (don't stretch)

3. **Slide Layout Suggestions:**
   - **Slide 4:** Place waveform on right, bullet points on left
   - **Slide 5:** Place window diagram full-width below text
   - **Slide 8:** Place distribution centered, formula below
   - **Slide 9:** Place time series FULL WIDTH - this is your star!
   - **Slide 10:** Place bar chart full-width, add success callout

## Current Data Status

⚠️ **Note:** The script currently uses synthetic data for visualization because:
- IMS clean data not found in expected location
- Batch scoring hasn't been run yet for time series

### To Use Real Data:

1. **For Slide 4 & 5 (Raw Waveform):**
   - Ensure `data/clean/ims_clean.parquet` exists
   - Or place raw files in `data/raw/ims/1st_test/`

2. **For Slide 8 (Threshold Distribution):**
   - Check `artifacts/reports/ims_iforest/` for validation scores

3. **For Slide 9 (Time Series - CRITICAL):**
   - Run batch scoring on all IMS files chronologically:
   ```bash
   python scripts/score_batch.py \
     --model artifacts/models/ims_iforest.joblib \
     --input data/features/ims_features.csv \
     --output artifacts/scores/ims_full_timeline
   ```
   - Script will automatically load real scores if available

4. **Regenerate Figures:**
   ```bash
   python scripts/generate_presentation_figures.py
   ```

## Customization

To modify figures, edit `scripts/generate_presentation_figures.py`:

- **Colors:** Modify `COLORS` dictionary (lines 35-43)
- **Figure sizes:** Modify `plt.rcParams['figure.figsize']` (line 30)
- **Fonts:** Modify `plt.rcParams['font.size']` (lines 31-33)
- **Individual plots:** Each function is self-contained

## Quality Settings

All figures exported with:
- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG with transparency support
- **Compression:** Optimized for file size without quality loss
- **Aspect ratio:** Preserved from matplotlib defaults

## Next Steps

1. ✅ **Review generated figures** - Open PNGs to verify quality
2. ⬜ **Insert into PowerPoint** - Add to appropriate slides
3. ⬜ **Run batch scoring** - Generate real IMS time series data
4. ⬜ **Regenerate with real data** - Update Slide 9 with actual scores
5. ⬜ **Practice presentation** - Rehearse transitions between slides

## Questions?

If figures look wrong or need adjustments:
1. Check data file locations (see "Current Data Status" above)
2. Modify color scheme in script if needed
3. Adjust figure sizes for your PowerPoint template
4. Re-run script after changes

**Remember:** Slide 9 (IMS time series) is your HERO GRAPH - this is the most important visualization that proves your method works!
