import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Juvenile\2019"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE --------
page_index = 483 # Page 504

# -------- COLUMNS --------
columns = [
    "SL","State",
    
    # Education
    "Illiterate",
    "Upto_Primary",
    "Primary_to_Matric",
    "Matric_to_Higher_Secondary",
    "Above_Higher_Secondary",
    "Total_Education",
    
    # Family Background
    "Living_with_Parents",
    "Living_with_Guardians",
    "Homeless",
    "Total_Family"
]

# -------- FINAL EXTRACTION FUNCTION --------
def extract_page(page):

    text = page.extract_text()

    # 🔥 KEY FIX (same working method)
    chunks = re.split(r'\n(?=\d+\s)', text)

    data = []
    expected = len(columns) - 2

    for chunk in chunks:

        chunk = chunk.strip()

        # Skip headers
        if any(k in chunk for k in ["STATES", "State/UT", "SL", "[", "Education", "Family"]):
            continue

        sl_match = re.match(r'^(\d+)', chunk)
        if not sl_match:
            continue

        sl = sl_match.group(1)

        # Remove SL
        content = chunk[len(sl):].strip()

        # Extract numbers
        nums = re.findall(r'\d+', content)

        # -------- STATE EXTRACTION --------
        num_match = re.search(r'\d', content)

        if num_match:
            state = content[:num_match.start()].strip()
        else:
            state = content.strip()

        state = re.sub(r'\s+', ' ', state)
        if not state and str(sl) == '31':
            state = 'D&N Haveli and Daman & Diu'

        # Fix length
        nums = (nums + ['0'] * expected)[:expected]

        data.append([sl, state] + nums)

    df = pd.DataFrame(data, columns=columns)

    # -------- CLEAN --------
    df["SL"] = df["SL"].astype(int)
    df = df.sort_values("SL").reset_index(drop=True)

    # Fix names
    df["State"] = df["State"].replace({
        "Daman": "Daman & Diu",
        "Diu": "Daman & Diu",
        "Andaman Nicobar Islands": "Andaman & Nicobar Islands"
    })

    # Convert numeric
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# -------- MAIN --------
with pdfplumber.open(pdf_path) as pdf:

    print("📄 Processing Page 504...")

    df = extract_page(pdf.pages[page_index])

    output_path = os.path.join(output_folder, "504_juvenile_education_family.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Saved: {output_path}")
    print(f"Rows extracted: {len(df)}")

    print("\n🔍 States:")
    print(df[["SL","State"]])

print("\n🎉 PAGE 504 EXTRACTION DONE")