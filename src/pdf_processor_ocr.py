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

            # 提取赛项名称和赛道 (根据用户反馈调整顺序：第一行是赛道，第二行是赛项名称)
            if len(lines_psm3) >= 1:
                # 第一行视为赛道
                potential_track_name = lines_psm3[0]
                # 赛道过滤条件 (简单判断)
                if len(potential_track_name) > 3 and ("赛" in potential_track_name or "组" in potential_track_name) and not re.search(r'\d{4}年', potential_track_name):
                    extracted_data["赛道"] = potential_track_name
            if len(lines_psm3) >= 2:
                # 第二行视为赛项名称
                potential_comp_name = lines_psm3[1]
                # 赛项名称过滤条件 (简单判断)
                if len(potential_comp_name) > 5 and ("赛" in potential_comp_name or "挑战" in potential_comp_name):
                     extracted_data["赛项名称"] = potential_comp_name
                # 如果第二行不像赛项名称，但第一行像赛道，尝试将第一行拆分 (OCR可能合并)
                # 这个逻辑可能需要调整或移除，取决于 OCR 实际效果
                # elif extracted_data["赛道"] and len(lines_psm3) == 1 and ' ' in extracted_data["赛道"]:
                #     parts = extracted_data["赛道"].split(' ', 1)
                #     if len(parts) == 2 and len(parts[1]) > 3:
                #         extracted_data["赛道"] = parts[0].strip()
                #         extracted_data["赛项名称"] = parts[1].strip()

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
        # 注意：这里我们在 full_text (保留换行符) 上搜索，而不是 full_text_cleaned

        # 提取报名时间
        # 模式解释:
        # (?:报名时间|作品提交时间|提交时间|报名阶段|时间安排|报名起止|提交起止) - 匹配各种可能的关键字
        # \s*[:：]\s* - 匹配关键字后的冒号和可选空格/换行
        # (                            - 开始捕获组 (报名时间的具体内容)
        #   [^，。；(（]*?             - 非贪婪地匹配任何非结束标点符号的字符
        #   (?:                     - 开始日期模式部分
        #     (?:\d{4}\s*年)?\s*\d{1,2}\s*月\s*\d{1,2}\s*日? - 匹配 YYYY年MM月DD日 (年和日可选)
        #     (?:\s*(?:至|—|－|-|~|到)\s*(?:\d{4}\s*年)?\s*\d{1,2}\s*月\s*\d{1,2}\s*日?)? - 匹配可选的结束日期
        #     | \d{1,2}\s*月\s*\d{1,2}\s*日? - 或者只匹配 MM月DD日
        #     | \d{4}\s*年 - 或者只匹配 YYYY年 
        #   )                         - 结束日期模式部分
        #   [^，。；(（]*?             - 再次非贪婪匹配，直到结束标点
        # )                            - 结束捕获组
        # 修改后的正则，更灵活处理换行和空格，并捕获关键字后的相关文本块直到明显的分隔符或下一主题。
        # 修改后的正则，加入“报名起始时间”关键字
        registration_match = re.search(
            r'(报名时间|作品提交时间|提交时间|报名阶段|时间安排|报名起止|提交起止|报名起始时间)\s*[:：]\s*([\s\S]*?)(?=\n\s*\n|[\n\r](?:[二三四五六七八九十]|\d+\.|\(.)\)|(?:提交方式|官网|联系方式|评审方式|奖项设置))'\
            , full_text, re.IGNORECASE) # 忽略大小写

        registration_text = None
        if registration_match:
            # 从匹配到的文本块中提取最可能的日期描述
            block = registration_match.group(2).strip()
            # 优先查找包含年份的完整日期或范围
            # 优化后的 date_detail_match 正则表达式，更明确地处理范围
            date_detail_match = re.search(
                # Group 1: 完整范围 YYYY年MM月DD日 - (YYYY年)?MM月DD日
                r'(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?\s*(?:至|—|－|-|~|到)\s*(?:\d{4}\s*年)?\s*\d{1,2}\s*月\s*\d{1,2}\s*日?)' +\
                # Group 2: 单独日期 YYYY年MM月DD日
                r'|(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?)' +\
                # Group 3: 无年份范围 MM月DD日 - MM月DD日
                r'|(\d{1,2}\s*月\s*\d{1,2}\s*日?\s*(?:至|—|－|-|~|到)\s*\d{1,2}\s*月\s*\d{1,2}\s*日?)' +\
                # Group 4: 单独无年份日期 MM月DD日 (优先级最低)
                r'|(\d{1,2}\s*月\s*\d{1,2}\s*日?)'\
                , block)
            if date_detail_match:
                 # 选择第一个非空的匹配组
                 registration_text = next((g for g in date_detail_match.groups() if g), None)
                 # 尝试扩展捕获范围以包含附近的文本
                 start_index = block.find(registration_text)
                 end_index = start_index + len(registration_text)
                 # 向前回溯查找可能的前缀（如"即日起"）
                 prefix_search_area = block[:start_index]
                 prefix_match = re.search(r'(即日起|自发布之日)\s*([至|—|－|-|~|到])?\s*$', prefix_search_area)
                 if prefix_match:
                     registration_text = prefix_match.group(1) + (prefix_match.group(2) or '至') + registration_text
                 # 向后查找可能的后缀 (如 "止")
                 suffix_search_area = block[end_index:]
                 suffix_match = re.match(r'^\s*(止|截止)', suffix_search_area)
                 if suffix_match:
                     registration_text += suffix_match.group(1)
            else:
                 # 如果找不到精确日期，就取整个块，但限制长度避免太长
                 registration_text = block[:100] # 限制最大长度

        extracted_data["报名时间"] = registration_text

        # 提取官网
        # 模式解释：
        # (?:官网|官方网站|大赛网址|网站|网址) - 匹配关键字
        # \s*[:：]?\s* - 匹配可选的冒号和空格/换行
        # (https?:\/\/[^\s\"\'<>，。；)）]+) - 捕获 URL，直到空格或特定标点
        website_match = re.search(
            r'(官网|官方网站|大赛网址|网站|网址)\s*[:：]?\s*(https?:\/\/[^\s\"\'<>，。；)）]+)'\
            , full_text, re.IGNORECASE)
        if website_match:
            extracted_data["官网"] = website_match.group(2).strip()
        else:
            # 备用：直接查找看起来像官网的 URL (通常包含 contest, challenge, competition, edu 等关键字)
            # 这个正则比较宽松，可能会误匹配，但作为备用方案
            url_match = re.search(
                r'(https?:\/\/[\w\.\-\/]+\.(?:com|cn|org|net|edu)[\w\.\-\/]*)' + # 匹配基础URL结构
                r'(?=[\s，。；)）]|\n)' # URL后面应该是空格或标点
                , full_text)
            if url_match:
                 url = url_match.group(1).strip()
                 # 简单检查 URL 是否看起来像一个主页或竞赛页面
                 if len(url) > 10 and ('.gov' not in url and '.pdf' not in url) : # 过滤掉一些明显不是官网的链接
                    # 进一步确认：检查URL附近是否有相关关键词 (在 full_text 中检查，范围更大)
                    context_window_start = max(0, url_match.start() - 50)
                    context_window_end = min(len(full_text), url_match.end() + 50)
                    context_window = full_text[context_window_start:context_window_end]
                    if any(kw in context_window for kw in ["官网", "网站", "网址", "链接", "平台", "大赛", "竞赛", "报名"]):
                        extracted_data["官网"] = url

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