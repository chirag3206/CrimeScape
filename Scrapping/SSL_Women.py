import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Women\2019\SSL"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE CONFIG (266–271) --------
page_config = {

    # Page 266
    253: {
        "file": "266_dowry_itp.csv",
        "columns": [
            "SL","State",
            "Dowry_I","Dowry_V","Dowry_R",
            "ITP_Total_I","ITP_Total_V","ITP_Total_R",
            "Sec5_I","Sec5_V","Sec5_R",
            "Sec6_I","Sec6_V","Sec6_R"
        ]
    },

    # Page 267
    254: {
        "file": "267_itp_sections_domestic.csv",
        "columns": [
            "SL","State",
            "Sec7_I","Sec7_V","Sec7_R",
            "Sec8_I","Sec8_V","Sec8_R",
            "Other_ITP_I","Other_ITP_V","Other_ITP_R",
            "Domestic_I","Domestic_V","Domestic_R"
        ]
    },

    # Page 268
    255: {
        "file": "268_cyber_crime.csv",
        "columns": [
            "SL","State",
            "Cyber_Total_I","Cyber_Total_V","Cyber_Total_R",
            "Sec67_I","Sec67_V","Sec67_R",
            "Other_Cyber_I","Other_Cyber_V","Other_Cyber_R"
        ]
    },

    # Page 269
    256: {
        "file": "269_pocso_main.csv",
        "columns": [
            "SL","State",
            "POCSO_Total_I","POCSO_Total_V","POCSO_Total_R",
            "Child_Rape_I","Child_Rape_V","Child_Rape_R",
            "Sexual_Assault_I","Sexual_Assault_V","Sexual_Assault_R",
            "Harassment_I","Harassment_V","Harassment_R"
        ]
    },

    # Page 270
    257: {
        "file": "270_pocso_other.csv",
        "columns": [
            "SL","State",
            "Pornography_I","Pornography_V","Pornography_R",
            "POCSO_Other_I","POCSO_Other_V","POCSO_Other_R",
            "Sec377_I","Sec377_V","Sec377_R"
        ]
    },

    # Page 271
    258: {
        "file": "271_total_sll_ipc.csv",
        "columns": [
            "SL","State",
            "Indecent_I","Indecent_V","Indecent_R",
            "Total_SLL_I","Total_SLL_V","Total_SLL_R",
            "Total_IPC_SLL_I","Total_IPC_SLL_V","Total_IPC_SLL_R"
        ]
    }
}

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


# -------- EXTRACTION FUNCTION --------
def extract_page(page, config):

    text = page.extract_text()
    if not text:
        return pd.DataFrame()

    lines = merge_lines(text.split("\n"))
    data = []

    for line in lines:

        if any(k in line for k in ["STATES", "State/UT", "SL", "[", "Act"]):
            continue

        # SL extraction
        sl_match = re.match(r'^(\d+)', line)
        if not sl_match:
            continue

        sl = sl_match.group(1)
        rest = line[len(sl):].strip()

        # Extract numbers
        nums = re.findall(r'\d+\.\d+|\d+', rest)
        expected = len(config["columns"]) - 2

        # Fix mismatch
        nums = (nums + ['0'] * expected)[:expected]

        # Extract state name
        state = re.split(r'\d', rest, maxsplit=1)[0].strip()
        if not state and str(sl) == '31':
            state = 'D&N Haveli and Daman & Diu'

        data.append([sl, state] + nums)

    df = pd.DataFrame(data, columns=config["columns"])

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

print("\n🎉 PAGES 266–271 EXTRACTED SUCCESSFULLY")