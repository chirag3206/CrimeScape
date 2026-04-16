import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Juvenile\2019\IPC"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE CONFIG (480–488) --------
page_config = {

    # 460
    459: {
        "file": "480_human_body_crimes_1.csv",
        "columns": [
            "SL","State",
            "Murder",
            "Culpable_Homicide_Not_Murder",
            "Death_By_Negligence",
            "Dowry_Deaths",
            "Abetment_Suicide"
        ]
    },

    # 461
    460: {
        "file": "481_human_body_crimes_2.csv",
        "columns": [
            "SL","State",
            "Attempt_Murder",
            "Attempt_Culpable_Homicide",
            "Attempt_Suicide",
            "Miscarriage_Infanticide_Foeticide",
            "Hurt",
            "Wrongful_Restraint"
        ]
    },

    # 462
    461: {
        "file": "482_severe_crimes.csv",
        "columns": [
            "SL","State",
            "Assault_Women_Modesty",
            "Kidnapping_Abduction",
            "Human_Trafficking",
            "Unnatural_Offences",
            "Rape",
            "Attempt_Rape"
        ]
    },

    # 463
    462: {
        "file": "483_state_public_tranquility.csv",
        "columns": [
            "SL","State",
            "Offences_State",
            "Unlawful_Assembly",
            "Rioting",
            "Promoting_Enmity",
            "Affray"
        ]
    },

    # 464
    463: {
        "file": "484_property_crimes_1.csv",
        "columns": [
            "SL","State",
            "Theft",
            "Burglary",
            "Extortion_Blackmail",
            "Robbery",
            "Dacoity",
            "Attempt_Dacoity_Robbery"
        ]
    },

    # 465
    464: {
        "file": "485_property_crimes_2.csv",
        "columns": [
            "SL","State",
            "Preparation_Dacoity",
            "Criminal_Misappropriation",
            "Criminal_Breach_Trust",
            "Receiving_Stolen_Property",
            "Counterfeiting",
            "Forgery_Cheating"
        ]
    },

    # 466
    465: {
        "file": "486_misc_crimes_1.csv",
        "columns": [
            "SL","State",
            "Disobedience_Public_Servant",
            "Election_Offences",
            "Obscene_Books",
            "Rash_Driving",
            "Obscene_Acts_Public",
            "Religion_Offences"
        ]
    },

    # 467
    466: {
        "file": "487_misc_crimes_2.csv",
        "columns": [
            "SL","State",
            "Cheating_Impersonation",
            "Mischief",
            "Arson",
            "Criminal_Trespass",
            "Cruelty_Relatives"
        ]
    },

    # 468
    467: {
        "file": "488_final_juvenile_crimes.csv",
        "columns": [
            "SL","State",
            "Insult_Modesty",
            "Criminal_Intimidation",
            "Other_IPC",
            "Total_Cognizable"
        ]
    }
}

# -------- ROBUST EXTRACTION ENGINE --------
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

            nums = re.findall(r'\d+', full_text)

            while len(nums) < expected and j < len(lines):
                full_text += " " + lines[j]
                nums = re.findall(r'\d+', full_text)
                j += 1

            state = re.split(r'\d', full_text, maxsplit=1)[0].strip()
            if not state:
                if str(sl) == '31':
                    state = 'D&N Haveli and Daman & Diu'
                else:
                    i = j
                    continue

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

print("\n🎉 PAGES 480–488 EXTRACTED SUCCESSFULLY")