import pandas as pd
import os

# Load raw data
df = pd.read_csv("data/raw-data.csv")

# Clean RAM column
df['RAM'] = df['RAM'].astype(str).str.extract(r'(\d+)').astype(float)

# Extract screen width and height from 'Screen' column safely
screen_res = df['Screen'].str.extract(r'(?P<width>\d+)[xX](?P<height>\d+)')
df['Screen_Width'] = pd.to_numeric(screen_res['width'], errors='coerce')
df['Screen_Height'] = pd.to_numeric(screen_res['height'], errors='coerce')

# Drop rows with missing resolution info
df = df.dropna(subset=['Screen_Width', 'Screen_Height'])

# Save clean data
os.makedirs("data", exist_ok=True)
df.to_csv("data/clean.csv", index=False)
