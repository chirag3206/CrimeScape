import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Women\2019\IPC"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE CONFIG (256–265) --------
page_config = {

    # Page 256
    243: {
        "file": "256_murder_dowry_abetment.csv",
        "columns": [
            "SL","State",
            "Murder_Rape_I","Murder_Rape_V","Murder_Rape_R",
            "Dowry_Deaths_I","Dowry_Deaths_V","Dowry_Deaths_R",
            "Abet_Suicide_I","Abet_Suicide_V","Abet_Suicide_R"
        ]
    },

    # Page 257
    244: {
        "file": "257_misc_acid_cruelty.csv",
        "columns": [
            "SL","State",
            "Miscarriage_I","Miscarriage_V","Miscarriage_R",
            "Acid_Attack_I","Acid_Attack_V","Acid_Attack_R",
            "Attempt_Acid_I","Attempt_Acid_V","Attempt_Acid_R",
            "Cruelty_I","Cruelty_V","Cruelty_R"
        ]
    },

    # Page 258
    245: {
        "file": "258_kidnap_total_363_364.csv",
        "columns": [
            "SL","State",
            "Kidnap_Total_I","Kidnap_Total_V","Kidnap_Total_R",
            "Sec363_I","Sec363_V","Sec363_R",
            "Sec364_I","Sec364_V","Sec364_R"
        ]
    },

    # Page 259
    246: {
        "file": "259_ransom_366_total_above18.csv",
        "columns": [
            "SL","State",
            "Ransom_I","Ransom_V","Ransom_R",
            "Marriage_Total_I","Marriage_Total_V","Marriage_Total_R",
            "Above18_I","Above18_V","Above18_R"
        ]
    },

    # Page 260
    247: {
        "file": "260_below18_procurement_import.csv",
        "columns": [
            "SL","State",
            "Below18_I","Below18_V","Below18_R",
            "Procurement_I","Procurement_V","Procurement_R",
            "Importation_I","Importation_V","Importation_R"
        ]
    },

    # Page 261
    248: {
        "file": "261_others_trafficking_selling.csv",
        "columns": [
            "SL","State",
            "Others_I","Others_V","Others_R",
            "Trafficking_I","Trafficking_V","Trafficking_R",
            "Selling_I","Selling_V","Selling_R"
        ]
    },

    # Page 262
    249: {
        "file": "262_buying_rape.csv",
        "columns": [
            "SL","State",
            "Buying_I","Buying_V","Buying_R",
            "Rape_Total_I","Rape_Total_V","Rape_Total_R",
            "Rape_18_I","Rape_18_V","Rape_18_R",
            "Rape_Below18_I","Rape_Below18_V","Rape_Below18_R"
        ]
    },

    # Page 263
    250: {
        "file": "263_attempt_rape.csv",
        "columns": [
            "SL","State",
            "Attempt_Total_I","Attempt_Total_V","Attempt_Total_R",
            "Attempt_18_I","Attempt_18_V","Attempt_18_R",
            "Attempt_Below18_I","Attempt_Below18_V","Attempt_Below18_R"
        ]
    },

    # Page 264
    251: {
        "file": "264_assault_modesty.csv",
        "columns": [
            "SL","State",
            "Assault_Total_I","Assault_Total_V","Assault_Total_R",
            "Assault_18_I","Assault_18_V","Assault_18_R",
            "Assault_Below18_I","Assault_Below18_V","Assault_Below18_R"
        ]
    },

    # Page 265
    252: {
        "file": "265_insult_total_crime.csv",
        "columns": [
            "SL","State",
            "Insult_Total_I","Insult_Total_V","Insult_Total_R",
            "Insult_18_I","Insult_18_V","Insult_18_R",
            "Insult_Below18_I","Insult_Below18_V","Insult_Below18_R",
            "Total_Crime_I","Total_Crime_V","Total_Crime_R"
        ]
    }
}

# -------- REGEX --------
pattern = re.compile(r'^(\d+)\s+([A-Za-z0-9 ,&\-\(\)\.]+?)\s+([\d\s\.]+)$')

# -------- MERGE LINES --------
def merge_lines(lines):
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\d+\s+', line):
            if buffer:
                merged.append(buffer)
            buffer = line
        else:
            buffer += " " + line

    if buffer:
        merged.append(buffer)

    return merged


def extract_page(page, config):

    text = page.extract_text()
    if not text:
        return pd.DataFrame()

    raw_lines = text.split("\n")

    # -------- STEP 1: CLEAN + MERGE --------
    merged_lines = []
    buffer = ""

    for line in raw_lines:
        line = line.strip()

        if not line:
            continue

        # New row starts
        if re.match(r'^\d+\s', line):
            if buffer:
                merged_lines.append(buffer)
            buffer = line
        else:
            buffer += " " + line

    if buffer:
        merged_lines.append(buffer)

    # -------- STEP 2: EXTRACT --------
    data = []

    for line in merged_lines:

        # Skip headers
        if any(k in line for k in ["STATES", "State/UT", "SL", "[", "IPC"]):
            continue

        # Extract SL
        sl_match = re.match(r'^(\d+)', line)
        if not sl_match:
            continue

        sl = sl_match.group(1)

        # Remove SL
        rest_line = line[len(sl):].strip()

        # Extract ALL numbers
        nums = re.findall(r'\d+\.\d+|\d+', rest_line)

        expected = len(config["columns"]) - 2

        # -------- CRITICAL FIX --------
        if len(nums) < expected:
            # Try to recover by splitting differently
            parts = rest_line.split()
            nums = [p for p in parts if re.match(r'\d+\.\d+|\d+', p)]

        # Trim/pad
        nums = (nums + ['0'] * expected)[:expected]

        # Extract state name = text before first number
        state_match = re.split(r'\d', rest_line, maxsplit=1)
        state = state_match[0].strip()
        if not state and str(sl) == '31':
            state = 'D&N Haveli and Daman & Diu'

        row = [sl, state] + nums
        data.append(row)

    df = pd.DataFrame(data, columns=config["columns"])

    # Convert numeric
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# -------- MAIN LOOP --------
with pdfplumber.open(pdf_path) as pdf:

    for page_index, config in page_config.items():

        print(f"\n📄 Processing Page {page_index+1}...")

        df = extract_page(pdf.pages[page_index], config)

        if df.empty:
            print("❌ No data extracted")
            continue

        output_path = os.path.join(output_folder, config["file"])
        df.to_csv(output_path, index=False)

        print(f"✅ Saved: {output_path}")
        print(f"Rows: {len(df)}")

print("\n🎉 ALL PAGES (256–265) EXTRACTED SUCCESSFULLY")