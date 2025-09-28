```markdown
# Pb-PEC-Calibration-Framework  
Weighted-least-squares calibration & LOD/LOQ calculator for Pb²⁺ PEC-sensor data.

## Citation  
MD Rayhan & Bingqian Liu (2025).  
*Pb²⁺ Biosensor Calibration Script*.  
Zenodo http://doi.org/10.5281/zenodo.17212395

## Installation  
```bash
git clone https://github.com/RAY186B/Pb-PEC-Calibration-Framework.git
cd Pb-PEC-Calibration-Framework
pip install -r requirements.txt
```

Usage  

```bash
# quick sanity check with synthetic data
python calibrate.py --demo

# real calibration
python calibrate.py your_data.csv --unit ng/mL --outdir results
```

Run `python calibrate.py -h` for all options.

Input CSV (minimal)  

Sample	Added_ng_mL	PEC_found_ng_mL	RSD_percent	
Tap water	0	0.001	4.5	
Tap water	1	1.05	3.2	

Outputs  
- `calibration_results.csv` – full statistics  
- `simple_results.csv` – slope, LOD, LOQ, R²  
- `*/calibration_plot.png` – one plot per sample

License  
MIT

```
