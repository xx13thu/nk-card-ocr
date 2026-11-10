import requests
import json
import re
from datetime import datetime
import time
import difflib
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

# ==========================
# 1. 基础配置区域
# ==========================
png_base_url = "https://prd-content.static.game.nogikoi.jp/assets/img/card/mypage/"
jpg_base_url = "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/"

# --- 从 GitHub Secrets 获取机密信息 ---
BOT_TOKEN = os.environ.get("BOT_TOKEN", "你的TOKEN")
CHANNEL_CHAT_ID = os.environ.get("CHANNEL_CHAT_ID", "你的ID")

# --- 日文汉字到英文罗马音的完美映射字典 ---
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
    
    # 兼容常见的日文异体字/OCR容错
    "海边朱莉": "KaibeAkari", "川崎桜": "KawasakiSakura"
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

# === 初始化 PaddleOCR 稳定版 (指定 lang='japan') ===
print("正在加载 PaddleOCR 日文模型...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='japan', show_log=False)

# --- 状态持久化管理 ---
STATE_FILE = "scan_state.json"
default_start_points = {
    '170': 14785, '181': 14785, '270': 14791, '281': 14791,
    '370': 14694, '381': 14694, '150': 7933, '161': 7933,
    '250': 7953, '261': 7953, '350': 7942, '361': 7942
}

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default_start_points

def save_state(state_dict):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=4)

start_points = load_state()
last_valid_urls = {prefix: None for prefix in start_points}
sent_urls = set()
invalid_counts = {prefix: 0 for prefix in start_points}
max_invalid = 20
completed_categories = set()

# ==========================
# 2. 核心功能函数
# ==========================
def send_photo_to_telegram(photo_url, caption, thread_id=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    payload = {"chat_id": CHANNEL_CHAT_ID, "photo": photo_url, "caption": caption}
    if thread_id is not None:
        payload["message_thread_id"] = thread_id
        
    for attempt in range(2):
        try:
            response = requests.post(url, data=payload, timeout=15)
            if response.status_code == 200:
                print(f"[成功] 推送: {photo_url}")
                return True
            else:
                print(f"[警告] 推送失败: {response.text}")
        except Exception as e:
            print(f"[错误] 网络异常: {e}")
        time.sleep(3)
    return False

def download_image(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except:
        return False

def paddle_ocr_jp(image_url):
    """
    专门针对日文大字的提取策略，完全舍弃底部模糊英文
    """
    try:
        # 1. 下载图片到内存
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        
        image_bytes = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # 2. 物理裁剪：保留下方 30%，左右全部保留
        h, w = img.shape[:2]
        cropped_img = img[int(h * 0.7):h, :]
        
        # 3. 适度放大 (2倍即可)，不使用任何破坏性二值化，保留自带阴影
        enlarged = cv2.resize(cropped_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        
        # 4. 执行日文 OCR
        result = ocr_engine.ocr(gray, cls=True)
        if not result or result[0] is None:
            return "LowConf", ""

        texts_jp = []
        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            
            # 阈值放宽到 0.3，宁可错杀不放过
            if conf < 0.3: 
                continue
                
            text = text.replace(" ", "").strip()
            # 过滤掉纯英文和数字，只保留日文汉字和假名
            if not re.fullmatch(r"[A-Za-z0-9]+", text):
                # 剔除混入其中的英文字母
                clean_jp = re.sub(r'[A-Za-z0-9]', '', text)
                if clean_jp:
                    texts_jp.append(clean_jp)

        # 把所有提取到的日文拼接在一起
        raw_jp_text = "".join(texts_jp)
        
        # 5. --- 核心：模糊匹配映射到英文 ---
        if raw_jp_text:
            valid_jp_keys = list(JP_TO_EN_MAP.keys())
            # 容错率 cutoff=0.35。比如把"海邊朱莉"错认成"海辺朱利"，依然能精准纠正回来
            matches = difflib.get_close_matches(raw_jp_text, valid_jp_keys, n=1, cutoff=0.35)
            if matches:
                matched_jp_name = matches[0]
                en_tag = JP_TO_EN_MAP[matched_jp_name]
                print(f"[{image_url}] 识别到日文: '{raw_jp_text}' -> 纠正为: '{matched_jp_name}' -> Tag: #{en_tag}")
                return en_tag, matched_jp_name
                
        return "LowConf", ""

    except Exception as e:
        print(f"[{image_url}] 解析发生异常: {e}")
        return "OCR_Error", ""

# ==========================
# 3. 主轮询任务
# ==========================
def download_task():
    global start_points
    while True:
        has_new_card = False 
        for prefix in list(start_points.keys()):
            if prefix in completed_categories: continue

            current_number = start_points[prefix]
            num_str = f"{prefix}{current_number:04d}" if current_number < 10000 else f"{prefix}{current_number:05d}"
            png_url = f"{png_base_url}{num_str}.png"
            jpg_url = f"{jpg_base_url}{num_str}.jpg"

            exists = download_image(png_url)

            if exists:
                has_new_card = True
                invalid_counts[prefix] = 0
                star = '⭐5' if prefix[1] == '5' else '⭐6' if prefix[1] == '6' else '⭐7' if prefix[1] == '7' else '⭐8'

                print(f"发现新卡，调用 PaddleOCR 识别日文: {jpg_url}")
                # 使用专门的日文识别映射引擎
                en_tag, jp_text = paddle_ocr_jp(jpg_url) 

                # --- 独立推送逻辑 ---
                target_thread_id = TOPIC_MAP.get(en_tag, DEFAULT_TOPIC_ID)
                
                # 1. 先推送 PNG
                caption = f"#{en_tag}\n\n{star}\n\n{png_url}"
                if len(caption) > 1000: caption = caption[:1000] + "..."
                send_photo_to_telegram(png_url, caption, thread_id=target_thread_id)
                
                time.sleep(1) # 缓冲间隔防止 TG 限流
                
                # 2. 紧接着推送 JPG
                send_photo_to_telegram(jpg_url, caption, thread_id=target_thread_id)

                start_points[prefix] += 1
                save_state(start_points) 
                
            else:
                invalid_counts[prefix] += 1
                start_points[prefix] += 1
                save_state(start_points)

            # 无效连续判定及回退机制
            if invalid_counts[prefix] >= max_invalid:
                completed_categories.add(prefix)
                start_points[prefix] -= max_invalid
                save_state(start_points) 
                print(f"[{prefix}] 连续 {max_invalid} 次无效，已回退序号至 {start_points[prefix]}。")

            time.sleep(1) # 本地 OCR 速度快，保留极短休眠即可
                
        if len(completed_categories) >= len(start_points):
            print("所有分类遍历完毕，任务结束。")
            break
            
        if not has_new_card: time.sleep(1)

def main():
    print(f"{datetime.now()} - 乃木恋 Bot 启动，使用 PaddleOCR 纯日文识别映射引擎...")
    download_task()

if __name__ == "__main__":
    main()