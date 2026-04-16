import pdfplumber
import pandas as pd
import re

pdf_path = r"D:\College\Projects\PRJ-3\Crimes\NCRB_Files\19_HT.pdf"

def extract_table_14_2(page):
    data = []
    pattern = re.compile(r'^(?:(\d+)\s+)?([a-zA-Z\s&()]+?)\s+((?:\d+\s*)+)$')
    
    lines = page.extract_text().split("\n")
    
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            sno, state, rest = match.groups()
            nums = rest.split()
            if len(nums) == 9:
                row = [sno if sno else "", state.strip()] + nums
                data.append(row)
    
    df = pd.DataFrame(data, columns=[
        "SNo","State",
        "Male<18","Female<18","Total<18",
        "Male>18","Female>18","Total>18",
        "MaleTotal","FemaleTotal","GrandTotal"
    ])
    
    return df

def extract_table_14_3(page):
    data = []
    pattern = re.compile(r'^(?:(\d+)\s+)?([a-zA-Z\s&()]+?)\s+((?:\d+\s*)+)$')
    
    lines = page.extract_text().split("\n")
    
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            sno, state, rest = match.groups()
            nums = rest.split()
            if len(nums) == 9:
                row = [sno if sno else "", state.strip()] + nums
                data.append(row)
    
    df = pd.DataFrame(data, columns=[
        "SNo","State",
        "Male<18","Female<18","Total<18",
        "Male>18","Female>18","Total>18",
        "MaleTotal","FemaleTotal","GrandTotal"
    ])
    
    return df


def extract_table_14_4(page1, page2):
    data = {}

    pattern = re.compile(r'^(?:(\d+)\s+)?([a-zA-Z\s&()]+?)\s+((?:\d+\s*)+)$')

    # -------- PAGE 1 (Indian, Sri Lanka, Nepal) --------
    lines = page1.extract_text().split("\n")

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            sno, state, rest = match.groups()
            state = state.strip()
            nums = rest.split()

            if len(nums) == 9:
                data[state] = {
                    "SNo": sno if sno else "",
                    "State": state,
                    "Indian_Male": nums[0],
                    "Indian_Female": nums[1],
                    "Indian_Total": nums[2],
                    "SriLanka_Male": nums[3],
                    "SriLanka_Female": nums[4],
                    "SriLanka_Total": nums[5],
                    "Nepal_Male": nums[6],
                    "Nepal_Female": nums[7],
                    "Nepal_Total": nums[8]
                }

    # -------- PAGE 2 (Bangladesh, Others, Total) --------
    lines = page2.extract_text().split("\n")

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            sno, state, rest = match.groups()
            state = state.strip()
            nums = rest.split()

            if len(nums) == 9 and state in data:
                data[state].update({
                    "Bangladesh_Male": nums[0],
                    "Bangladesh_Female": nums[1],
                    "Bangladesh_Total": nums[2],
                    "Others_Male": nums[3],
                    "Others_Female": nums[4],
                    "Others_Total": nums[5],
                    "Total_Male": nums[6],
                    "Total_Female": nums[7],
                    "Total_Total": nums[8]
                })

    df = pd.DataFrame(list(data.values()))

    return df


def extract_table_14_5(page3, page4):
    data = {}

    pattern = re.compile(r'^(?:(\d+)\s+)?([a-zA-Z\s&()]+?)\s+((?:\d+\s*)+)$')

    # -------- PAGE 3 --------
    lines = page3.extract_text().split("\n")

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            sno, state, rest = match.groups()
            state = state.strip()
            nums = rest.split()

            if len(nums) == 6:
                data[state] = {
                    "SNo": sno if sno else "",
                    "State": state,
                    "Forced Labour": nums[0],
                    "Sexual Exploitation": nums[1],
                    "Other Sexual Exploitation": nums[2],
                    "Domestic Servitude": nums[3],
                    "Forced Marriage": nums[4],
                    "Petty Crimes": nums[5]
                }

    # -------- PAGE 4 --------
    lines = page4.extract_text().split("\n")

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            sno, state, rest = match.groups()
            state = state.strip()
            nums = rest.split()

            if len(nums) == 6 and state in data:
                data[state].update({
                    "Child Pornography": nums[0],
                    "Begging": nums[1],
                    "Drug Peddling": nums[2],
                    "Removal of Organs": nums[3],
                    "Other Reasons": nums[4],
                    "Total Persons": nums[5]
                })

    df = pd.DataFrame(list(data.values()))

    return df


with pdfplumber.open(pdf_path) as pdf:
    
    df1 = extract_table_14_2(pdf.pages[0])
    df1.to_csv(r"D:\College\Projects\PRJ-3\Crimes\CSV\Human Traficking\2019\table_14_2.csv", index=False)
    print(f"Table 14.2: {len(df1)} rows")
    
    df2 = extract_table_14_3(pdf.pages[1])
    df2.to_csv(r"D:\College\Projects\PRJ-3\Crimes\CSV\Human Traficking\2019\table_14_3.csv", index=False)
    print(f"Table 14.3: {len(df2)} rows")
    
    df3 = extract_table_14_4(pdf.pages[2], pdf.pages[3])
    df3.to_csv(r"D:\College\Projects\PRJ-3\Crimes\CSV\Human Traficking\2019\table_14_4.csv", index=False)
    print(f"Table 14.4: {len(df3)} rows")
    
    df4 = extract_table_14_5(pdf.pages[4], pdf.pages[5])
    df4.to_csv(r"D:\College\Projects\PRJ-3\Crimes\CSV\Human Traficking\2019\table_14_5.csv", index=False)
    print(f"Table 14.5: {len(df4)} rows")

print("All tables extracted successfully")