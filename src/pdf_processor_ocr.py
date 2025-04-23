import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import pandas as pd # 添加 pandas 用于写入 Excel

# --- 配置 ---
# 确保 Tesseract OCR 的路径正确配置 (如果不在系统 PATH 中)
# 例如: pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # macOS/Linux
# 例如: pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' # Windows

PDF_DIR = "/Users/tcw/Desktop/SmartAgent/data/raw"  # PDF 文件所在的目录 (相对于脚本的位置)
OUTPUT_EXCEL = "/Users/tcw/Desktop/SmartAgent/results/result_1.xlsx" # 输出 Excel 文件路径
# -------------

def extract_pdf_information(pdf_path):
    """
    从 PDF 文件中提取所需信息，结合 OCR (首页) 和全文文本提取。

    Args:
        pdf_path (str): PDF 文件的路径。

    Returns:
        dict: 包含提取信息的字典，键为表头名称。
               如果某项信息找不到，对应值为 None。
    """
    extracted_data = {
        "赛项名称": None,
        "赛道": None,
        "发布时间": None,
        "报名时间": None,
        "组织单位": None,
        "官网": None
    }
    full_text = ""

    try:
        doc = fitz.open(pdf_path)

        # --- 1. 提取整个文档的文本 --- 
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text += page.get_text("text") + "\n" # 使用 fitz 提取文本

        # 清理提取的文本，合并多余的换行和空格
        full_text_cleaned = re.sub(r'\s+', ' ', full_text.replace('\n', ' ')).strip()

        # --- 2. OCR 处理第一页，提取标题、组织单位、发布时间 --- 
        if len(doc) > 0:
            page = doc[0]
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # OCR for Title/Track (PSM 3)
            text_psm3 = pytesseract.image_to_string(image, config=r'--oem 3 --psm 3 -l chi_sim')
            lines_psm3 = [line.strip() for line in text_psm3.split('\n') if line.strip()]

            if len(lines_psm3) >= 1:
                potential_comp_name = lines_psm3[0]
                if len(potential_comp_name) > 5 and ("赛" in potential_comp_name or "挑战" in potential_comp_name):
                     extracted_data["赛项名称"] = potential_comp_name
            if len(lines_psm3) >= 2:
                potential_track_name = lines_psm3[1]
                if len(potential_track_name) > 3 and ("赛" in potential_track_name or "组" in potential_track_name) and not re.search(r'\d{4}年', potential_track_name):
                    extracted_data["赛道"] = potential_track_name
                elif extracted_data["赛项名称"] and len(lines_psm3) == 1 and ' ' in extracted_data["赛项名称"]:
                    parts = extracted_data["赛项名称"].split(' ', 1)
                    if len(parts) == 2 and len(parts[1]) > 3:
                        extracted_data["赛项名称"] = parts[0].strip()
                        extracted_data["赛道"] = parts[1].strip()

            # OCR for Org/Date (PSM 6)
            text_psm6 = pytesseract.image_to_string(image, config=r'--oem 3 --psm 6 -l chi_sim')

            date_match = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', text_psm6)
            if not date_match:
                 date_match = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', text_psm3)

            if date_match:
                extracted_data["发布时间"] = date_match.group(0).replace(" ", "")

            lines_psm6 = [line.strip() for line in text_psm6.split('\n') if line.strip()]
            date_line_index = -1
            for i, line in enumerate(lines_psm6):
                if extracted_data["发布时间"] and extracted_data["发布时间"].replace("年", "").replace("月", "") in line.replace(" ", ""): # 查找包含日期的行
                    date_pattern_in_line = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', line)
                    if date_pattern_in_line and date_pattern_in_line.group(0).replace(" ", "") == extracted_data["发布时间"]:
                        date_line_index = i
                        break

            if date_line_index > 0:
                potential_unit = lines_psm6[date_line_index - 1]
                if len(potential_unit) > 3 and not potential_unit.isdigit():
                    extracted_data["组织单位"] = potential_unit
            elif lines_psm6:
                 keywords = ["中心", "委员会", "协会", "学会", "办公室", "组委会"]
                 unit_found_psm6 = False
                 for line in reversed(lines_psm6):
                     if any(kw in line for kw in keywords) and len(line) > 3:
                         extracted_data["组织单位"] = line
                         unit_found_psm6 = True
                         break
                 if not unit_found_psm6 and extracted_data["发布时间"] and lines_psm6[-1] != extracted_data["发布时间"]:
                    potential_unit = lines_psm6[-1]
                    if len(potential_unit) > 3 and not potential_unit.isdigit():
                        extracted_data["组织单位"] = potential_unit

            if not extracted_data["组织单位"]:
                lines_psm3_lower = [line.strip() for line in text_psm3.split('\n') if line.strip()]
                date_line_index_psm3 = -1
                for i, line in enumerate(lines_psm3_lower):
                    if extracted_data["发布时间"] and extracted_data["发布时间"].replace("年", "").replace("月", "") in line.replace(" ", ""): # 查找包含日期的行
                        date_pattern_in_line = re.search(r'(\d{4})\s*年\s*(\d{1,2})\s*月', line)
                        if date_pattern_in_line and date_pattern_in_line.group(0).replace(" ", "") == extracted_data["发布时间"]:
                            date_line_index_psm3 = i
                            break

                if date_line_index_psm3 > 0:
                    potential_unit = lines_psm3_lower[date_line_index_psm3 - 1]
                    if len(potential_unit) > 3 and not potential_unit.isdigit():
                        extracted_data["组织单位"] = potential_unit
                elif lines_psm3_lower:
                    keywords = ["中心", "委员会", "协会", "学会", "办公室", "组委会"]
                    for line in reversed(lines_psm3_lower):
                         if any(kw in line for kw in keywords) and len(line) > 3:
                            extracted_data["组织单位"] = line
                            break
                    if not extracted_data["组织单位"] and extracted_data["发布时间"] and lines_psm3_lower[-1] != extracted_data["发布时间"]:
                        potential_unit = lines_psm3_lower[-1]
                        if len(potential_unit) > 3 and not potential_unit.isdigit():
                           extracted_data["组织单位"] = potential_unit

        # --- 3. 从全文文本中提取报名时间和官网 --- 

        # 提取报名时间 (模式: 报名时间[::\s].*?([\\d]{4}年.*?日)) - 匹配到第一个日期
        # 更宽松的模式，匹配 "报名时间" 后的第一个日期范围或单个日期
        # (?:报名时间|提交时间|报名阶段|时间安排)[::\s]+([^，。；]*?(?:\\d{4}\\s*年.*?月.*?日|至|—|－|-|~)[^，。；]*)?
        # 匹配关键字后直到下一个标点符号的内容，并尝试从中找日期模式
        registration_match = re.search(
            r'(?:报名时间|作品提交时间|提交时间|报名阶段|时间安排)\s*[:\s]+([^，。；\n\(（]+(?:(?:\\d{4}\\s*年)?\\s*\\d{1,2}\\s*月\\s*\\d{1,2}\\s*日?\\s*(?:至|—|－|-|~|到)\\s*(?:\\d{4}\\s*年)?\\s*\\d{1,2}\\s*月\\s*\\d{1,2}\\s*日?|\\d{4}\\s*年\\s*\\d{1,2}\\s*月\\s*\\d{1,2}\\s*日?))',
            full_text_cleaned)
        if registration_match:
            extracted_data["报名时间"] = registration_match.group(1).strip()
        else: # 备用：尝试查找"报名起止"类的关键字
            registration_match_alt = re.search(r'(?:报名起止|提交起止)\s*[:\s]+([^，。；\n\(（]+)', full_text_cleaned)
            if registration_match_alt:
                 potential_time = registration_match_alt.group(1).strip()
                 # 确认包含日期信息
                 if re.search(r'\d{1,2}\s*月\s*\d{1,2}\s*日', potential_time):
                      extracted_data["报名时间"] = potential_time

        # 提取官网 (模式: 官网|网站[::\s].*?(https?://[\w\./-=&%#]+))
        website_match = re.search(r'(?:官网|官方网站|大赛网址|网站|网址)\s*[:\s]*(https?:\\/\\/[^,\.;\)）]+)', full_text_cleaned)
        if website_match:
            extracted_data["官网"] = website_match.group(1).strip()
        else: # 备用：直接查找 URL
             url_match = re.search(r'(https?:\\/\\/[\\w\\.\\-\\/]+\\.(?:com|cn|org|net|edu))[,\.;\)）]', full_text_cleaned)
             if url_match:
                 # 确认这个 URL 前面有相关字眼，避免误提取
                 search_window = full_text_cleaned[max(0, url_match.start()-30):url_match.start()]
                 if any(kw in search_window for kw in ["官网", "网站", "网址", "链接", "平台"]):
                     extracted_data["官网"] = url_match.group(1).strip()

        doc.close()

    except Exception as e:
        print(f"处理文件 '{pdf_path}' 时出错: {e}")

    # 清理结果中的潜在换行符和多余空格
    for key, value in extracted_data.items():
        if isinstance(value, str):
            extracted_data[key] = re.sub(r'\s+', ' ', value).strip()

    return extracted_data

def main():
    """主函数，遍历 PDF 目录，提取信息并保存到 Excel"""
    if not os.path.isdir(PDF_DIR):
        print(f"错误：找不到 PDF 目录 '{PDF_DIR}'")
        return

    print(f"开始处理目录 '{PDF_DIR}' 中的 PDF 文件...")

    all_extracted_data = [] # 用于存储所有 PDF 的提取结果

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf") and not filename.startswith('.'): # 忽略隐藏文件
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"--- 正在处理: {filename} ---")

            # 提取信息
            data = extract_pdf_information(pdf_path)
            all_extracted_data.append(data)

            # 打印当前文件提取结果
            print(f"  赛项名称: {data['赛项名称'] if data['赛项名称'] else '未能提取'}")
            print(f"  赛道:     {data['赛道'] if data['赛道'] else '未能提取'}")
            print(f"  组织单位: {data['组织单位'] if data['组织单位'] else '未能提取'}")
            print(f"  发布时间: {data['发布时间'] if data['发布时间'] else '未能提取'}")
            print(f"  报名时间: {data['报名时间'] if data['报名时间'] else '未能提取'}")
            print(f"  官网:     {data['官网'] if data['官网'] else '未能提取'}")
            print("-" * (len(filename) + 15)) # 分隔线

    # --- 将结果写入 Excel --- 
    if all_extracted_data:
        print(f"\n正在将提取的 {len(all_extracted_data)} 条记录写入到 '{OUTPUT_EXCEL}'...")
        # 确保 results 目录存在
        output_dir = os.path.dirname(OUTPUT_EXCEL)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df = pd.DataFrame(all_extracted_data)
        # 按照要求的列顺序排列
        column_order = ["赛项名称", "赛道", "发布时间", "报名时间", "组织单位", "官网"]
        df = df[column_order]

        try:
            df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl') # 需要安装 openpyxl: pip install openpyxl
            print("Excel 文件写入成功!")
        except Exception as e:
            print(f"写入 Excel 文件时出错: {e}")
    else:
        print("没有提取到任何数据，未生成 Excel 文件。")

if __name__ == "__main__":
    main() 