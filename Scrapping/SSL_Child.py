import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Children\2019\SSL"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE CONFIG --------
page_config = {

    # 379
    358: {
        "file": "375_pocso_sec4_6.csv",
        "columns": ["SL","State",
            "POCSO_Total_I","POCSO_Total_V","POCSO_Total_R",
            "Sec4_6_Total_I","Sec4_6_Total_V","Sec4_6_Total_R",
            "Girls_I","Girls_V","Girls_R",
            "Boys_I","Boys_V","Boys_R"]
    },

    # 380
    359: {
        "file": "376_pocso_sec8_10.csv",
        "columns": ["SL","State",
            "Sec8_10_Total_I","Sec8_10_Total_V","Sec8_10_Total_R",
            "Girls_I","Girls_V","Girls_R",
            "Boys_I","Boys_V","Boys_R"]
    },

    # 381
    360: {
        "file": "377_pocso_sec12.csv",
        "columns": ["SL","State",
            "Sec12_Total_I","Sec12_Total_V","Sec12_Total_R",
            "Girls_I","Girls_V","Girls_R",
            "Boys_I","Boys_V","Boys_R"]
    },

    # 382
    361: {
        "file": "378_pocso_sec14_15.csv",
        "columns": ["SL","State",
            "Sec14_15_Total_I","Sec14_15_Total_V","Sec14_15_Total_R",
            "Girls_I","Girls_V","Girls_R",
            "Boys_I","Boys_V","Boys_R"]
    },

    # 383
    362: {
        "file": "379_pocso_sec377.csv",
        "columns": ["SL","State",
            "Sec377_Total_I","Sec377_Total_V","Sec377_Total_R",
            "Girls_I","Girls_V","Girls_R",
            "Boys_I","Boys_V","Boys_R"]
    },

    # 384
    363: {
        "file": "380_pocso_17_22.csv",
        "columns": ["SL","State",
            "Sec17_22_Total_I","Sec17_22_Total_V","Sec17_22_Total_R",
            "Girls_I","Girls_V","Girls_R",
            "Boys_I","Boys_V","Boys_R"]
    },

    # 385
    364: {
        "file": "381_juvenile_justice.csv",
        "columns": ["SL","State",
            "JJ_Total_I","JJ_Total_V","JJ_Total_R",
            "Caretaker_I","Caretaker_V","Caretaker_R",
            "Other_I","Other_V","Other_R"]
    },

    # 386
    365: {
        "file": "382_itp_children.csv",
        "columns": ["SL","State",
            "ITP_Total_I","ITP_Total_V","ITP_Total_R",
            "Sec5_I","Sec5_V","Sec5_R",
            "Sec6_I","Sec6_V","Sec6_R",
            "Others_I","Others_V","Others_R"]
    },

    # 387
    366: {
        "file": "383_child_labour_marriage_organs.csv",
        "columns": ["SL","State",
            "Child_Labour_I","Child_Labour_V","Child_Labour_R",
            "Marriage_I","Marriage_V","Marriage_R",
            "Organs_I","Organs_V","Organs_R"]
    },

    # 388
    367: {
        "file": "384_cyber_children.csv",
        "columns": ["SL","State",
            "Cyber_Total_I","Cyber_Total_V","Cyber_Total_R",
            "Explicit_I","Explicit_V","Explicit_R",
            "Other_I","Other_V","Other_R"]
    },

    # 389
    368: {
        "file": "385_final_sll_children.csv",
        "columns": ["SL","State",
            "Other_SLL_I","Other_SLL_V","Other_SLL_R",
            "Total_SLL_I","Total_SLL_V","Total_SLL_R",
            "Total_All_I","Total_All_V","Total_All_R"]
    }
}

# -------- FINAL ROBUST ENGINE --------
def extract_page(page, config):

    text = page.extract_text()
    if not text:
        return pd.DataFrame()

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    data = []

    i = 0
    expected = len(config["columns"]) - 2

    while i < len(lines):

        line = lines[i]

        if any(k in line for k in ["STATES", "State/UT", "SL", "[", "POCSO"]):
            i += 1
            continue

        if re.match(r'^\d+\s', line):

            sl = re.match(r'^(\d+)', line).group(1)
            rest = line[len(sl):].strip()

            full_text = rest
            j = i + 1

            nums = re.findall(r'\d+\.\d+|\d+', full_text)

            # 🔥 Keep collecting lines
            while len(nums) < expected and j < len(lines):
                full_text += " " + lines[j]
                nums = re.findall(r'\d+\.\d+|\d+', full_text)
                j += 1

            state = re.split(r'\d', full_text, maxsplit=1)[0].strip()
            if not state and str(sl) == '31':
                state = 'D&N Haveli and Daman & Diu'

            nums = (nums + ['0'] * expected)[:expected]

            data.append([sl, state] + nums)

            i = j
        else:
            i += 1

    df = pd.DataFrame(data, columns=config["columns"])

    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# -------- MAIN --------
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

print("\n🎉 PAGES 375–385 EXTRACTED SUCCESSFULLY")