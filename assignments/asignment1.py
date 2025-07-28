import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading ---
print("--- Data Loading ---")
try:
    df = pd.read_csv(r'C:\Users\Pushpalatha A\OneDrive\Desktop\Samson\Wine_Quality_Data.csv')
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
except FileNotFoundError:
    print("Error: 'Wine_Quality_Data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- 2. Preliminary Cleaning ---
print("\n--- Preliminary Cleaning ---")

# Check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# A common strategy for missing values in numerical columns is to fill with median/mean
# or drop rows/columns. For this dataset, let's assume dropping rows with any missing data
# if they exist, to keep it simple.
initial_rows = df.shape[0]
df.dropna(inplace=True)
rows_after_dropna = df.shape[0]
if initial_rows > rows_after_dropna:
    print(f"\nDropped {initial_rows - rows_after_dropna} rows with missing values.")
else:
    print("\nNo missing values found or dropped (all data was clean).")

# Verify data types (df.info() already gives a good overview,
# but we can explicitly check if any column needs specific conversion if not inferred correctly)
# For this dataset, pandas should correctly infer most types.

# --- 3. Simple Transformations ---
print("\n--- Simple Transformations ---")

# Create a new feature: 'sulfur_dioxide_ratio'
# This ratio can indicate the proportion of free SO2 to total SO2.
# We'll handle division by zero by setting the ratio to 0 if total_sulfur_dioxide is 0.
df['sulfur_dioxide_ratio'] = df.apply(
    lambda row: row['free_sulfur_dioxide'] / row['total_sulfur_dioxide'] if row['total_sulfur_dioxide'] != 0 else 0,
    axis=1
)
print("Created a new feature: 'sulfur_dioxide_ratio'. First 5 rows with new feature:")
print(df[['free_sulfur_dioxide', 'total_sulfur_dioxide', 'sulfur_dioxide_ratio']].head())

# --- 4. Insightful Visualizations ---
print("\n--- Generating Visualizations ---")

plt.style.use('seaborn-v0_8-darkgrid') # Set a nice style for plots

# Visualization 1: Distribution of Wine Quality
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Distribution of Wine Quality Ratings')
plt.xlabel('Quality (0-10)')
plt.ylabel('Number of Wines')
plt.xticks(rotation=0)
plt.show()

# Visualization 2: Relationship between Alcohol Content and Quality
plt.figure(figsize=(12, 7))
sns.boxplot(x='quality', y='alcohol', data=df, palette='magma')
plt.title('Wine Quality by Alcohol Content')
plt.xlabel('Quality (0-10)')
plt.ylabel('Alcohol Content (%)')
plt.show()

# Visualization 3: Categorical Breakdown - Quality by Wine Color
plt.figure(figsize=(12, 7))
sns.countplot(x='quality', hue='color', data=df, palette={'red': 'salmon', 'white': 'lightblue'})
plt.title('Wine Quality Distribution by Color')
plt.xlabel('Quality (0-10)')
plt.ylabel('Number of Wines')
plt.legend(title='Wine Color')
plt.show()

print("\nAnalysis complete. Visualizations displayed.")
