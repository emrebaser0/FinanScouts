import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import pandas as pd

# Load the Excel file
data = pd.read_excel("Data/Processed_durus_01.04.2024-30.04.2024.xlsx")

# Split columns A:L (first 12 columns) and M:FP (remaining columns for scrap data)
fixed_columns = data.iloc[:, :12]
scrap_columns = data.iloc[:, 12:]

# Initialize an empty DataFrame to store the result
result = pd.DataFrame()

# Iterate through each row of the dataset
for index, row in data.iterrows():
    for col in scrap_columns.columns:
        scrap_reason = col
        scrap_count = row[col]
        # Only add rows where scrap count is not NaN or 0
        if pd.notna(scrap_count) and scrap_count != 0:
            new_row = row[:12].tolist() + [scrap_reason, scrap_count]
            result = pd.concat([
                result,
                pd.DataFrame([new_row], columns=list(fixed_columns.columns) + ['Hurda Tanımı', 'Hurda Adedi'])
            ], ignore_index=True)

# Save the result to a new Excel file
output_path = 'Data/processed_scrap_data.xlsx'
result.to_excel(output_path, index=False)
output_path

# Convert the Excel file to CSV
output_csv_path = 'Data/EB_Processed_durus_01.04.2024-30.04.2024.csv'
result.to_csv(output_csv_path, index=False)




