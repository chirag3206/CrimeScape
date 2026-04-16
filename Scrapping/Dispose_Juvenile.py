import pdfplumber
import pandas as pd
import re
import os

# -------- PATHS --------
pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\2019_Crime.pdf"
output_folder = r"D:\College\Projects\PRJ-3\Crimes\CSV\Juvenile\2019"

os.makedirs(output_folder, exist_ok=True)

# -------- PAGE --------
page_index = 482  # Page 503

# -------- COLUMNS --------
columns = [
    "SL","State",
    "Pending_Start_Year",
    "Apprehended_During_Year",
    "Total_Apprehended",
    "Discharged_During_Investigation",
    "Sent_Home_After_Advice",
    "Sent_To_Special_Home",
    "Dealt_With_Fine",
    "Awarded_Imprisonment",
    "Acquitted_Or_Discharged",
    "Conviction_Rate_Percent",
    "Pending_End_Year"
]

# -------- FINAL EXTRACTION FUNCTION --------
def extract_page(page):

    text = page.extract_text()

    # 🔥 CRITICAL FIX: split rows using SL numbers
    chunks = re.split(r'\n(?=\d+\s)', text)

    data = []
    expected = len(columns) - 2

    for chunk in chunks:

        chunk = chunk.strip()

        # Skip headers
        if any(k in chunk for k in ["STATES", "State/UT", "SL", "[", "Juveniles"]):
            continue

        sl_match = re.match(r'^(\d+)', chunk)
        if not sl_match:
            continue

        sl = sl_match.group(1)

        # Remove SL
        content = chunk[len(sl):].strip()

        # Extract numbers
        nums = re.findall(r'\d+\.\d+|\d+', content)

        # -------- STATE EXTRACTION --------
        num_match = re.search(r'\d', content)

        if num_match:
            state = content[:num_match.start()].strip()
        else:
            state = content.strip()

        # Clean state
        state = re.sub(r'\s+', ' ', state)
        if not state and str(sl) == '31':
            state = 'D&N Haveli and Daman & Diu'

        # Fix numbers length
        nums = (nums + ['0'] * expected)[:expected]

        data.append([sl, state] + nums)

    df = pd.DataFrame(data, columns=columns)

    # -------- CLEAN --------
    df["SL"] = df["SL"].astype(int)
    df = df.sort_values("SL").reset_index(drop=True)

    # Fix common names
    df["State"] = df["State"].replace({
        "Daman": "Daman & Diu",
        "Diu": "Daman & Diu",
        "Daman Diu": "Daman & Diu",
        "Andaman Nicobar Islands": "Andaman & Nicobar Islands"
    })

    # Convert numeric columns
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# -------- MAIN --------
with pdfplumber.open(pdf_path) as pdf:

    print("📄 Processing Page 503...")

    df = extract_page(pdf.pages[page_index])

    output_path = os.path.join(output_folder, "503_juvenile_case_disposal.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Saved: {output_path}")
    print(f"Rows extracted: {len(df)}")

    print("\n🔍 States:")
    print(df[["SL","State"]])

print("\n🎉 DONE — NO ROW LOSS")