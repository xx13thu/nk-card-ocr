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

# --- 映射字典 ---
JP_TO_EN_MAP = {
    "伊藤理々杏": "ItoRiria", "岩本蓮加": "IwamotoRenka", "吉田綾乃クリスティー": "YoshidaAyanoChristie", 
    "遠藤さくら": "EndoSakura", "賀喜遥香": "KakiHaruka", "金川紗耶": "KanagawaSaya", 
    "柴田柚菜": "ShibataYuna", "田村真佑": "TamuraMayu", "筒井あやめ": "TsutsuiAyame", 
    "黒見明香": "KuromiHaruka", "林瑠奈": "HayashiRuna", "弓木奈於": "YumikiNao",
    
    "五百城茉央": "IokiMao", "池田瑛紗": "IkedaTeresa", "一ノ瀬美空": "IchinoseMiku", 
    "井上和": "InoueNagi", "岡本姫奈": "OkamotoHina", "小川彩": "OgawaAya", 
    "奥田いろは": "OkudaIroha", "川﨑桜": "KawasakiSakura", "菅原咲月": "SugawaraSatsuki", 
    "冨里奈央": "TomisatoNao", "中西アルノ": "NakanishiAruno",
    
    "愛宕心響": "AtagoKokone", "大越ひなの": "OkoshiHinano", "小津玲奈": "OzuReina", 
    "海邊朱莉": "KaibeAkari", "川端晃菜": "KawabataHina", "鈴木佑捺": "SuzukiYuuna", 
    "瀬戸口心月": "SetoguchiMitsuki", "長嶋凛桜": "NagashimaRio", "増田三莉音": "MasudaMirine", 
    "森平麗心": "MorihiraUrumi", "矢田萌華": "YadaMoeka",
}

TOPIC_MAP = {
    "ItoRiria": 2, "IwamotoRenka": 4, "YoshidaAyanoChristie": 6, "EndoSakura": 8,
    "KakiHaruka": 10, "KanagawaSaya": 12, "ShibataYuna": 14, "TamuraMayu": 16,
    "TsutsuiAyame": 18, "KuromiHaruka": 20, "HayashiRuna": 22, "YumikiNao": 24,
    "IokiMao": 26, "IkedaTeresa": 28, "IchinoseMiku": 30, "InoueNagi": 32,
    "OkamotoHina": 34, "OgawaAya": 36, "OkudaIroha": 38, "KawasakiSakura": 40,
    "SugawaraSatsuki": 42, "TomisatoNao": 44, "NakanishiAruno": 46, "AtagoKokone": 48,
    "OkoshiHinano": 50, "OzuReina": 52, "KaibeAkari": 54, "KawabataHina": 56,
    "SuzukiYuuna": 58, "SetoguchiMitsuki": 60, "NagashimaRio": 62, "MasudaMirine": 64,
    "MorihiraUrumi": 66, "YadaMoeka": 68,
}
DEFAULT_TOPIC_ID = None

# --- 全新状态持久化管理 ---
STATE_FILE = "scan_state.json"
raw_default_points = {
    '170': 14785, '181': 14785, '270': 14791, '281': 14791,
    '370': 14694, '381': 14694, '150': 7933, '161': 7933,
    '250': 7953, '261': 7953, '350': 7942, '361': 7942
}

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 无感升级：旧格式自动转为新带 gaps 的格式
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
    try:
        # 1. 获取图片
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        
        # === 核心修复：使用轻量级 Pillow 裁剪底部 30% ===
        img = Image.open(io.BytesIO(resp.content))
        w, h = img.size
        # 裁剪坐标 (left, upper, right, lower)
        cropped_img = img.crop((0, int(h * 0.7), w, h))
        
        # 将裁剪后的大字图片转为 Base64
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 2. 构造请求发送给 Gemini
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": "提取图片里的乃木坂46成员的日文名字。如果图片上有多个名字，请把它们都提取出来，并用英文逗号 ',' 隔开。请只输出日文汉字，绝不要包含拼音、说明或其他任何标点符号。"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }],
            "generationConfig": {"temperature": 0.0}
        }

        res = requests.post(api_url, json=payload, timeout=30)
        res_json = res.json()
        
        # 拦截并打印 API 错误（如额度超限、Key失效）
        if "error" in res_json:
            print(f"[API 报错] {res_json['error']['message']}")
            return []

        if "candidates" in res_json and len(res_json["candidates"]) > 0:
            raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            raw_text = raw_text.replace("，", ",") 
            
            # === 新增调试打印：看看 Gemini 到底输出了什么 ===
            print(f"[{image_url}] Gemini 原始返回: '{raw_text}'")
            
            names_list = [name.strip().replace("\n", "").replace(" ", "") for name in raw_text.split(",")]
            
            en_tags = []
            valid_jp_keys = list(JP_TO_EN_MAP.keys())
            
            for raw_jp_text in names_list:
                if not raw_jp_text: continue
                matches = difflib.get_close_matches(raw_jp_text, valid_jp_keys, n=1, cutoff=0.3)
                if matches:
                    en_tags.append(JP_TO_EN_MAP[matches[0]])
                else:
                    print(f"[{image_url}] 字典匹配失败，未在字典中找到相似项: '{raw_jp_text}'")
            
            if en_tags:
                en_tags = list(dict.fromkeys(en_tags)) 
                return en_tags 
                
        return []
        
    except Exception as e:
        print(f"[{image_url}] Gemini 解析发生异常: {e}")
        return []

def process_and_send_card(prefix, png_url, jpg_url):
    """
    统一的图文处理分发中心：负责多成员多 Topic 的独立分发推送
    """
    star = '⭐5' if prefix[1] == '5' else '⭐6' if prefix[1] == '6' else '⭐7' if prefix[1] == '7' else '⭐8'
    en_tags = gemini_vision_ocr(jpg_url)
    
    if not en_tags:
        caption = f"#LowConf\n\n{star}\n\n{png_url}"
        send_photo_to_telegram(png_url, caption[:1000], thread_id=DEFAULT_TOPIC_ID)
    else:
        # 将所有识别到的名字拼接成 Tag (如：#EndoSakura #KakiHaruka)
        tags_str = " ".join([f"#{tag}" for tag in en_tags])
        caption = f"{tags_str}\n\n{star}\n\n{png_url}"
        
        # 收集涉及到的所有群组 Topic ID (去重，防发两遍)
        target_thread_ids = []
        for tag in en_tags:
            tid = TOPIC_MAP.get(tag, DEFAULT_TOPIC_ID)
            if tid not in target_thread_ids:
                target_thread_ids.append(tid)
                
        # 向所有涉及到的成员专属 Topic 分别推送一次
        for tid in target_thread_ids:
            send_photo_to_telegram(png_url, caption[:1000], thread_id=tid)
            
            # --- [想要保留 JPG 推送功能？取消下方两行的注释即可] ---
            time.sleep(1)
            send_photo_to_telegram(jpg_url, caption[:1000], thread_id=tid)
            
            time.sleep(1) # 防止 TG 限流

# ==========================
# 3. 主轮询任务
# ==========================
def download_task():
    global start_points
    
    # ============================================
    # 阶段一：优先排查历史遗留跳号 (Gaps)
    # ============================================
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
                
                # 补录成功，从跳号名单中彻底除名
                del gaps[gap_str]
                save_state(start_points)
            else:
                # 依然没找到，重试次数加 1
                gaps[gap_str] += 1
                if gaps[gap_str] >= 10:
                    print(f"[{prefix}] 序号 {gap_num} 连续 10 次未能找回，确认为永久空号。")
                    del gaps[gap_str]
                save_state(start_points)
            time.sleep(1)

    # ============================================
    # 阶段二：常规向前探索 (Round-Robin)
    # ============================================
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
                
                # 【核心】：只有成功找到新卡，才将之前的空缺填入补录名单
                head = start_points[prefix]["head"]
                for missing_num in range(head, curr_num):
                    start_points[prefix]["gaps"][str(missing_num)] = 0
                    print(f"[{prefix}] 嗅探到跳号间隙: {missing_num}，已加入回溯通缉名单。")
                
                # 游标真正往前挪
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
    print(f"{datetime.now()} - 乃木恋 Bot 启动 (终极整合多发版，搭载跳号回溯引擎)...")
    download_task()

if __name__ == "__main__":
    main()