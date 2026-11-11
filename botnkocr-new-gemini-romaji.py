import requests
import json
import base64
import re
from datetime import datetime
import time
import difflib
import os
import io
from PIL import Image

# ==========================
# 1. 基础配置区域
# ==========================
png_base_url = "https://prd-content.static.game.nogikoi.jp/assets/img/card/mypage/"
jpg_base_url = "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/"

# --- GitHub Secrets ---
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHANNEL_CHAT_ID = os.environ.get("CHANNEL_CHAT_ID", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

TOPIC_MAP = {
    "ItoRiria": 2, "IwamotoRenka": 4, "YoshidaAyanoChristie": 6, "EndoSakura": 8,
    "KakiHaruka": 10, "KanagawaSaya": 12, "ShibataYuna": 14, "TamuraMayu": 16,
    "TsutsuiAyame": 18, "KuromiHaruka": 20, "HayashiRuna": 22, "YumikiNao": 24,
    
    "IokiMao": 26, "IkedaTeresa": 28, "IchinoseMiku": 30, "InoueNagi": 32,
    "OkamotoHina": 34, "OgawaAya": 36, "OkudaIroha": 38, "KawasakiSakura": 40,
    "SugawaraSatsuki": 42, "TomisatoNao": 44, "NakanishiAruno": 46,
    
    "AtagoKokone": 48, "OkoshiHinano": 50, "OzuReina": 52, "KaibeAkari": 54, 
    "KawabataHina": 56, "SuzukiYuuna": 58, "SetoguchiMitsuki": 60, "NagashimaRio": 62, 
    "MasudaMirine": 64, "MorihiraUrumi": 66, "YadaMoeka": 68
}

# [神级容错]: 建立一个全小写、无空格的映射字典
LOWER_TO_REAL_MAP = {k.lower(): k for k in TOPIC_MAP.keys()}
DEFAULT_TOPIC_ID = None

# --- 状态持久化管理 ---
STATE_FILE = "scan_state.json"
raw_default_points = {
    '170': 14810, '181': 14810, '270': 14816, '281': 14816,
    '370': 14720, '381': 14720, '150': 7937, '161': 7937,
    '250': 7957, '261': 7957, '350': 7946, '361': 7946
}

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for k, v in data.items():
                if isinstance(v, int):
                    data[k] = {"head": v, "gaps": {}}
            return data
    return {k: {"head": v, "gaps": {}} for k, v in raw_default_points.items()}

def save_state(state_dict):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=4)

start_points = load_state()
max_invalid = 20

# ==========================
# 2. 核心功能函数
# ==========================
def send_photo_to_telegram(photo_url, caption, thread_id=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    payload = {"chat_id": CHANNEL_CHAT_ID, "photo": photo_url, "caption": caption}
    if thread_id is not None: payload["message_thread_id"] = thread_id
    
    for attempt in range(2):
        try:
            resp = requests.post(url, data=payload, timeout=15)
            if resp.status_code == 200: return True
        except: pass
        time.sleep(3)
    return False

def download_image(url):
    try:
        return requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).status_code == 200
    except:
        return False

def gemini_vision_ocr(image_url):
    max_retries = 3 
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(image_url, timeout=15)
            resp.raise_for_status()
            
            img = Image.open(io.BytesIO(resp.content))
            w, h = img.size
            cropped_img = img.crop((0, int(h * 0.77), w, h))
            
            buffered = io.BytesIO()
            cropped_img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{
                    "parts": [
                        # === 提取指令 ===
                        {
                            "text": "请提取图片底部的英文文本，可能是乃木坂46成员的罗马音姓名，也可能是活动标题。如果有多个成员姓名，请用英文逗号 ',' 隔开。如果是活动标题，直接输出。"
                        },
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                    ]
                }],
                "generationConfig": {"temperature": 0.0}
            }

            res = requests.post(api_url, json=payload, timeout=30)
            res_json = res.json()
            
            # 服务器防超载重试逻辑
            if "error" in res_json:
                error_msg = res_json['error'].get('message', '未知错误')
                print(f"[{image_url}] 第 {attempt + 1} 次请求报错: {error_msg}")
                if "high demand" in error_msg.lower() or res_json['error'].get('code') in [429, 503]:
                    if attempt < max_retries - 1:
                        print("检测到服务器拥挤，代码暂停 10 秒后重试...")
                        time.sleep(10)
                        continue 
                return []

            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                parts_list = res_json["candidates"][0]["content"]["parts"]
                raw_text = "".join([part.get("text", "") for part in parts_list]).strip()
                raw_text = raw_text.replace("，", ",") 
                names_list = [name.strip() for name in raw_text.split(",")]
                
                en_tags = []
                
                for raw_name in names_list:
                    if not raw_name: continue
                    
                    # 容错处理：大模型提取的可能是 "Hayashi Runa"，去掉空格全小写变成 "hayashiruna"
                    processed_name = raw_name.replace(" ", "").lower()
                    
                    # 在全小写字典里进行高容错模糊匹配
                    matches = difflib.get_close_matches(processed_name, list(LOWER_TO_REAL_MAP.keys()), n=1, cutoff=0.5)
                    
                    if matches:
                        # 找到了！把它转回我们在 TOPIC_MAP 里设定的标准驼峰大小写，比如 "HayashiRuna"
                        standard_name = LOWER_TO_REAL_MAP[matches[0]]
                        en_tags.append(standard_name)
                    else:
                        # 没找到（比如限定卡的英文副标题），去掉标点后首字母大写原样输出
                        safe_tag = re.sub(r'[^A-Za-z]+', '', raw_name).capitalize()
                        if safe_tag:
                            en_tags.append(safe_tag)
                
                if en_tags:
                    en_tags = list(dict.fromkeys(en_tags)) 
                    return en_tags 
                    
            return []
            
        except Exception as e:
            print(f"[{image_url}] 第 {attempt + 1} 次执行代码发生异常: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                return []

def process_and_send_card(prefix, png_url, jpg_url):
    star = '⭐5' if prefix[1] == '5' else '⭐6' if prefix[1] == '6' else '⭐7' if prefix[1] == '7' else '⭐8'
    en_tags = gemini_vision_ocr(jpg_url)
    
    if not en_tags:
        send_photo_to_telegram(png_url, f"#LowConf\n\n{star}\n\n{png_url}", thread_id=DEFAULT_TOPIC_ID)
        # 失败保底时，把 JPG 顺便发出来帮你认卡
        time.sleep(1)
        send_photo_to_telegram(jpg_url, f"#LowConf\n\n{star}\n\n{jpg_url}", thread_id=DEFAULT_TOPIC_ID)
    else:
        tags_str = " ".join([f"#{tag}" for tag in en_tags])
        caption_png = f"{tags_str}\n\n{star}\n\n{png_url}"
        caption_jpg = f"{tags_str}\n\n{star}\n\n{png_url}"
        
        target_thread_ids = []
        for tag in en_tags:
            tid = TOPIC_MAP.get(tag, DEFAULT_TOPIC_ID)
            if tid not in target_thread_ids:
                target_thread_ids.append(tid)
                
        for tid in target_thread_ids:
            send_photo_to_telegram(png_url, caption_png[:1000], thread_id=tid)
            time.sleep(0.5)
            send_photo_to_telegram(jpg_url, caption_jpg[:1000], thread_id=tid)
            time.sleep(1) 

# ==========================
# 3. 主轮询任务
# ==========================
def download_task():
    global start_points
    
    print("=== 开始排查历史跳号 ===")
    for prefix, state_data in start_points.items():
        gaps = state_data.get("gaps", {})
        for gap_str in list(gaps.keys()):
            gap_num = int(gap_str)
            num_str = f"{prefix}{gap_num:04d}" if gap_num < 10000 else f"{prefix}{gap_num:05d}"
            png_url = f"{png_base_url}{num_str}.png"
            jpg_url = f"{jpg_base_url}{num_str}.jpg"
            
            if download_image(png_url):
                print(f"[{prefix}] 成功捕获迟来的卡面: {jpg_url}")
                process_and_send_card(prefix, png_url, jpg_url)
                
                del gaps[gap_str]
                save_state(start_points)
            else:
                gaps[gap_str] += 1
                if gaps[gap_str] >= 20:
                    print(f"[{prefix}] 序号 {gap_num} 连续 20 次未能找回，确认为永久空号。")
                    del gaps[gap_str]
                save_state(start_points)
            time.sleep(1)

    print("=== 开始向前探索新卡面 ===")
    current_scan_pointers = {p: start_points[p]["head"] for p in start_points}
    invalid_counts = {p: 0 for p in start_points}
    completed_categories = set()
    
    while True:
        has_new_card = False 
        for prefix in list(start_points.keys()):
            if prefix in completed_categories: continue
            
            curr_num = current_scan_pointers[prefix]
            num_str = f"{prefix}{curr_num:04d}" if curr_num < 10000 else f"{prefix}{curr_num:05d}"
            png_url = f"{png_base_url}{num_str}.png"
            jpg_url = f"{jpg_base_url}{num_str}.jpg"

            if download_image(png_url):
                has_new_card = True
                invalid_counts[prefix] = 0
                
                head = start_points[prefix]["head"]
                for missing_num in range(head, curr_num):
                    start_points[prefix]["gaps"][str(missing_num)] = 0
                    print(f"[{prefix}] 嗅探到跳号间隙: {missing_num}，已加入回溯通缉名单。")
                
                start_points[prefix]["head"] = curr_num + 1
                
                print(f"发现新卡: {jpg_url}")
                process_and_send_card(prefix, png_url, jpg_url)
                
                current_scan_pointers[prefix] += 1
                save_state(start_points) 
            else:
                invalid_counts[prefix] += 1
                current_scan_pointers[prefix] += 1
                
            if invalid_counts[prefix] >= max_invalid:
                completed_categories.add(prefix)
                print(f"[{prefix}] 连续 {max_invalid} 次未命中，触及当日前沿尽头。")
                
            time.sleep(1)
                
        if len(completed_categories) >= len(start_points):
            print("所有类别探索完毕，任务完美收官。")
            break
            
        if not has_new_card: time.sleep(1)

def main():
    print(f"{datetime.now()} - 乃木恋 Bot 启动 (罗马音提取版)...")
    download_task()

if __name__ == "__main__":
    main()
