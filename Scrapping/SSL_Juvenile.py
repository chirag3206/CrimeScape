import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Juvenile\2019\SSL"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE CONFIG --------
page_config = {

    # Page 469 (index 468) - Columns [3]-[6]
    468: {
        "file": "489_women_children_related.csv",
        "columns": [
            "SL", "State",
            "Dowry_Prohibition_Act",
            "Immoral_Traffic_Act",
            "POCSO_Act",
            "Juvenile_Justice_Act"
        ]
    },

    # Page 470 (index 469) - Columns [7]-[11]
    469: {
        "file": "490_state_arms_acts.csv",
        "columns": [
            "SL", "State",
            "SC_Atrocities_Act",
            "ST_Atrocities_Act",
            "Damage_Public_Property_Act",
            "Unlawful_Activities_Act",
            "Arms_Act"
        ]
    },

    # Page 471 (index 470) - Columns [12]-[17]
    470: {
        "file": "491_it_finance_acts.csv",
        "columns": [
            "SL", "State",
            "Information_Technology_Act",
            "Lotteries_Act",
            "Mines_Minerals_Act",
            "Prohibition_Act",
            "Excise_Act",
            "NDPS_Act"
        ]
    },

    # Page 472 (index 471) - Columns [18]-[24]
    471: {
        "file": "492_drugs_environment.csv",
        "columns": [
            "SL", "State",
            "Forest_Act",
            "Wildlife_Protection_Act",
            "Environment_Protection_Act",
            "Tobacco_Act",
            "Noise_Pollution",
            "Foreigners_Act",
            "Passport_Act"
        ]
    },

    # Page 473 (index 472) - Columns [25]-[29]
    472: {
        "file": "493_railways_food_regulatory.csv",
        "columns": [
            "SL", "State",
            "Railways_Act",
            "Essential_Commodities_Act",
            "Food_Safety_Act",
            "Electricity_Act",
            "Gambling_Act"
        ]
    },

    # Page 474 (index 473) - Columns [30]-[35]
    473: {
        "file": "494_final_sll_summary.csv",
        "columns": [
            "SL", "State",
            "Motor_Vehicle_Act",
            "City_Police_Acts",
            "Defacement_Public_Property",
            "Other_State_Local_Acts",
            "Other_SLL_Crimes",
            "Total_SLL_Crimes"
        ]
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

        if any(k in line for k in ["STATES", "State/UT", "SL", "[", "Act"]):
            i += 1
            continue

        if re.match(r'^\d+\s', line):

            sl = re.match(r'^(\d+)', line).group(1)
            rest = line[len(sl):].strip()

            full_text = rest
            j = i + 1

            nums = re.findall(r'\d+', full_text)

            # 🔥 keep consuming lines
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

print("\n🎉 PAGES 489–494 EXTRACTED SUCCESSFULLY")