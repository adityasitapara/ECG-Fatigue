import pandas as pd

path = '../Data/MimicDemo'
patients = pd.read_csv(f'{path}/PATIENTS.csv')
icustays = pd.read_csv(f'{path}/ICUSTAYS.csv')
charts   = pd.read_csv(f'{path}/CHARTEVENTS.csv', low_memory=False)
d_items  = pd.read_csv(f'{path}/D_ITEMS.csv')

print(f"✅ Patients: {len(patients)}")
print(f"✅ ICU stays: {len(icustays)}")
print(f"✅ Chart events: {len(charts)}")
print(f"✅ Item dict entries: {len(d_items)}")

# Check actual column names first
print("\nD_ITEMS columns:", d_items.columns.tolist())

# Find label column dynamically (handles uppercase or lowercase)
label_col = [c for c in d_items.columns if c.lower() == 'label'][0]
heart_rate = d_items[d_items[label_col].str.contains('Heart Rate', na=False)]
print("\nHeart Rate items found:")
print(heart_rate[[c for c in d_items.columns if c.lower() in ['itemid','label']]].head())