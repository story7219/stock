import os
import shutil
import xml.etree.ElementTree as ET

# 1. CORPCODE.xml에서 ETN corp_code 추출
etn_corp_codes = set()
xml_path = "CORPCODE.xml"
tree = ET.parse(xml_path)
root = tree.getroot()
for corp in root.findall('.//list'):
    corp_code = corp.find('corp_code').text if corp.find('corp_code') is not None else ""
    corp_name = corp.find('corp_name').text if corp.find('corp_name') is not None else ""
    sector = corp.find('sector').text if corp.find('sector') is not None else ""
    product = corp.find('product').text if corp.find('product') is not None else ""
    text = f"{corp_name} {sector} {product}".upper()
    if "ETN" in text:
        etn_corp_codes.add(corp_code)

# 2. ETN corp_code 폴더만 삭제
del_count = 0
base_dir = "dart_historical_data"
for corp_code in etn_corp_codes:
    folder_path = os.path.join(base_dir, corp_code)
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"삭제: {folder_path}")
        del_count += 1

print(f"총 삭제된 ETN 폴더 수: {del_count}")
