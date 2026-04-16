import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Children\2019\IPC"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE CONFIG  --------
page_config = {

    # Page 370
    349: {
        "file": "366_murder_children.csv",
        "columns": [
            "SL","State",
            "Murder_I","Murder_V","Murder_R",
            "Murder_Rape_I","Murder_Rape_V","Murder_Rape_R",
            "Other_Murder_I","Other_Murder_V","Other_Murder_R",
            "Abet_Suicide_I","Abet_Suicide_V","Abet_Suicide_R"
        ]
    },

    # Page 371
    350: {
        "file": "367_attempt_infanticide.csv",
        "columns": [
            "SL","State",
            "Attempt_Murder_I","Attempt_Murder_V","Attempt_Murder_R",
            "Infanticide_I","Infanticide_V","Infanticide_R",
            "Foeticide_I","Foeticide_V","Foeticide_R",
            "Exposure_I","Exposure_V","Exposure_R"
        ]
    },

    # Page 372
    351: {
        "file": "368_hurt_kidnap.csv",
        "columns": [
            "SL","State",
            "Simple_Hurt_I","Simple_Hurt_V","Simple_Hurt_R",
            "Grievous_Hurt_I","Grievous_Hurt_V","Grievous_Hurt_R",
            "Kidnap_Total_I","Kidnap_Total_V","Kidnap_Total_R",
            "Sec363_I","Sec363_V","Sec363_R"
        ]
    },

    # Page 373
    352: {
        "file": "369_missing_other_begging.csv",
        "columns": [
            "SL","State",
            "Missing_I","Missing_V","Missing_R",
            "Other_Kidnap_I","Other_Kidnap_V","Other_Kidnap_R",
            "Begging_I","Begging_V","Begging_R"
        ]
    },

    # Page 374
    353: {
        "file": "370_murder_ransom_marriage.csv",
        "columns": [
            "SL","State",
            "Sec364_I","Sec364_V","Sec364_R",
            "Ransom_I","Ransom_V","Ransom_R",
            "Marriage_I","Marriage_V","Marriage_R"
        ]
    },

    # Page 375
    354: {
        "file": "371_procurement_import_other.csv",
        "columns": [
            "SL","State",
            "Procurement_I","Procurement_V","Procurement_R",
            "Import_I","Import_V","Import_R",
            "Other_Kidnap_I","Other_Kidnap_V","Other_Kidnap_R"
        ]
    },

    # Page 376
    355: {
        "file": "372_trafficking_selling.csv",
        "columns": [
            "SL","State",
            "Trafficking_I","Trafficking_V","Trafficking_R",
            "Selling_Total_I","Selling_Total_V","Selling_Total_R",
            "Boys_I","Boys_V","Boys_R",
            "Girls_I","Girls_V","Girls_R"
        ]
    },

    # Page 377
    356: {
        "file": "373_buying_rape_attempt.csv",
        "columns": [
            "SL","State",
            "Buying_Total_I","Buying_Total_V","Buying_Total_R",
            "Boys_I","Boys_V","Boys_R",
            "Girls_I","Girls_V","Girls_R",
            "Rape_I","Rape_V","Rape_R",
            "Attempt_I","Attempt_V","Attempt_R"
        ]
    },

    # Page 378
    357: {
        "file": "374_final_children.csv",
        "columns": [
            "SL","State",
            "Assault_I","Assault_V","Assault_R",
            "Insult_I","Insult_V","Insult_R",
            "Other_IPC_I","Other_IPC_V","Other_IPC_R",
            "Total_I","Total_V","Total_R"
        ]
    }
}

# -------- FINAL ROBUST EXTRACTION --------
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

        if any(k in line for k in ["STATES", "State/UT", "SL", "[", "IPC"]):
            i += 1
            continue

        if re.match(r'^\d+\s', line):

            sl = re.match(r'^(\d+)', line).group(1)
            rest = line[len(sl):].strip()

            full_text = rest
            j = i + 1

            nums = re.findall(r'\d+\.\d+|\d+', full_text)

            # 🔥 Keep consuming lines until numbers complete
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

print("\n🎉 PAGES 366–374 EXTRACTED SUCCESSFULLY")