import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Women\2019"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE CONFIG --------
page_index = 259 # Page 272

columns = [
    "SL","State",
    "Cases_Reported",

    "Below_6",
    "6_to_12",
    "12_to_16",
    "16_to_18",

    "Total_Child",

    "18_to_30",
    "30_to_45",
    "45_to_60",
    "60_plus",

    "Total_Adult",
    "Total_Victims"
]

# -------- MERGE LINES --------
def merge_lines(lines):
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\d+\s', line):
            if buffer:
                merged.append(buffer)
            buffer = line
        else:
            buffer += " " + line

    if buffer:
        merged.append(buffer)

    return merged


# -------- EXTRACTION --------
def extract_page(page):

    text = page.extract_text()
    if not text:
        return pd.DataFrame()

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    data = []

    i = 0
    expected = len(columns) - 2

    while i < len(lines):

        line = lines[i]

        # Skip headers
        if any(k in line for k in ["STATES", "State/UT", "SL", "[", "Victims"]):
            i += 1
            continue

        # Check if row starts
        if re.match(r'^\d+\s', line):

            sl = re.match(r'^(\d+)', line).group(1)

            # Remove SL
            rest = line[len(sl):].strip()

            full_text = rest

            # 👇 KEEP ADDING NEXT LINES UNTIL NUMBERS COMPLETE
            j = i + 1
            nums = re.findall(r'\d+\.\d+|\d+', full_text)

            while len(nums) < expected and j < len(lines):
                full_text += " " + lines[j]
                nums = re.findall(r'\d+\.\d+|\d+', full_text)
                j += 1

            # Extract state name safely
            state = re.split(r'\d', full_text, maxsplit=1)[0].strip()
            if not state and str(sl) == '31':
                state = 'D&N Haveli and Daman & Diu'

            # Fix numbers
            nums = (nums + ['0'] * expected)[:expected]

            data.append([sl, state] + nums)

            i = j  # move pointer forward

        else:
            i += 1

    df = pd.DataFrame(data, columns=columns)

    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# -------- MAIN --------
with pdfplumber.open(pdf_path) as pdf:

    print("📄 Processing Page 272...")

    df = extract_page(pdf.pages[page_index])

    if df.empty:
        print("❌ No data extracted")
    else:
        output_path = os.path.join(output_folder, "272_age_group_rape.csv")
        df.to_csv(output_path, index=False)

        print(f"✅ Saved: {output_path}")
        print(f"Rows: {len(df)}")

print("\n🎉 PAGE 272 EXTRACTED SUCCESSFULLY")