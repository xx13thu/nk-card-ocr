import requests
import json
import re
from datetime import datetime
import time
import difflib
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR # === 引入 PaddleOCR ===

# ==========================
# 1. 基础配置区域
# ==========================
png_base_url = "https://prd-content.static.game.nogikoi.jp/assets/img/card/mypage/"
jpg_base_url = "https://prd-content.static.game.nogikoi.jp/assets/img/card/l/"

# === 从环境变量读取机密信息（配合 GitHub Secrets） ===
BOT_TOKEN = os.environ.get("BOT_TOKEN", "你本地测试用的TOKEN")
CHANNEL_CHAT_ID = os.environ.get("CHANNEL_CHAT_ID", "你本地测试用的ID")

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

# === 初始化 PaddleOCR (全局只初始化一次) ===
# lang='japan' 表示支持日文和英文，use_angle_cls 开启方向分类
print("正在加载 PaddleOCR 模型...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# === 状态持久化配置 ===
STATE_FILE = "state.json"
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
    # 此处保持你的原逻辑不变
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
        except Exception as e:
            pass
        time.sleep(3)
    return False

def download_image(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        return resp.status_code == 200
    except:
        return False

# === 替换为 PaddleOCR 版本的识别引擎 ===
def paddle_ocr(image_url):
    try:
        # 1. 下载图片到内存
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        
        image_bytes = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        # 2. 物理裁剪：保留下方 30%，左右全部保留
        h, w = img.shape[:2]
        cropped_img = img[int(h * 0.7):h, :]
        
        # 3. 放大 3 倍并转为灰度
        enlarged = cv2.resize(cropped_img, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        
        # ==========================================
        # 构建 3 种不同强度的预处理图像矩阵
        # ==========================================
        images_to_try = []
        
        # [方案 A] 基础灰度 (适用：远藤樱、深色背景、自带阴影)
        images_to_try.append(gray)
        
        # [方案 B] 强力伽马压暗 + 拉伸 (适用：海边朱莉、白底白字、极低对比度)
        # 将非纯白像素呈指数级压暗，剔除浅色皮肤和蕾丝的干扰
        gamma = 0.3 
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        darkened = cv2.LUT(gray, table)
        stretched = cv2.normalize(darkened, None, 0, 255, cv2.NORM_MINMAX) # 重新拉伸对比度
        images_to_try.append(stretched)
        
        # [方案 C] 高斯模糊 + 自适应二值化翻转 (适用：大越ひなの、复杂高频纹理背景)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 3)
        images_to_try.append(cv2.bitwise_not(binary))
        
        # ==========================================
        # 循环验证逻辑：只要中奖一次，立刻退出
        # ==========================================
        best_tag = "LowConf"
        
        for idx, process_img in enumerate(images_to_try):
            result = ocr_engine.ocr(process_img, cls=True)
            if not result or result[0] is None:
                continue

            texts_en = []
            blacklist = ['SSR', 'SR', 'UR', 'SUMMER']
            
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                conf = line[1][1]
                
                if conf < 0.3: 
                    continue
                    
                text = text.replace(" ", "").strip()
                if re.search(r"[A-Za-z]", text):
                    text = text.replace("l", "I")
                    clean_en = re.sub(r'[^a-zA-Z0-9]', '', text)
                    if clean_en and clean_en.upper() not in blacklist:
                        center_x = sum([p[0] for p in box]) / 4
                        texts_en.append((clean_en, center_x))
            
            if texts_en:
                # 排序拼接提取到的字母
                texts_en.sort(key=lambda item: item[1])
                sorted_words = [item[0] for item in texts_en]
                final_texts_en = list(dict.fromkeys(sorted_words))
                raw_tag = "".join(final_texts_en)
                
                # --- 核心：立刻调用模糊匹配查验字典 ---
                # 如果这个滤镜提取出来的乱码，能被字典救回来并精准匹配到 TOPIC_MAP 中的某个键
                matched_tag = fuzzy_match_tag(raw_tag, TOPIC_MAP)
                
                if matched_tag in TOPIC_MAP:
                    print(f"[{image_url}] 在第 {idx+1} 套图像预处理中完美识别: {matched_tag}")
                    return matched_tag, ""  # 识别成功，直接返回，不再浪费算力测试后续滤镜
                else:
                    # 如果这套滤镜也没认全，先暂存最好结果，留给下一个滤镜继续努力
                    best_tag = matched_tag
                    
        # 如果 3 套处理全跑完都没能在字典里找到绝对匹配项，返回最后一次得到的近似值
        return best_tag, ""

    except Exception as e:
        print(f"[{image_url}] 解析发生异常: {e}")
        return "OCR_Error", ""

def fuzzy_match_tag(raw_tag, valid_tags_dict):
    # 保持你原有的模糊匹配逻辑不变
    if raw_tag in ["LowConf", "OCR_Error"] or raw_tag in valid_tags_dict: return raw_tag
    raw_lower = raw_tag.lower()
    valid_lower_map = {k.lower(): k for k in valid_tags_dict.keys()}
    matches = difflib.get_close_matches(raw_lower, valid_lower_map.keys(), n=1, cutoff=0.5)
    return valid_lower_map[matches[0]] if matches else raw_tag

# ==========================
# 3. 主轮询任务
# ==========================
def download_task():
    global start_points
    while True:
        has_new_card = False 
        for prefix in list(start_points.keys()):
            if prefix in completed_categories:
                continue

            current_number = start_points[prefix]
            num_str = f"{prefix}{current_number:04d}" if current_number < 10000 else f"{prefix}{current_number:05d}"
            png_url = f"{png_base_url}{num_str}.png"
            jpg_url = f"{jpg_base_url}{num_str}.jpg"

            exists = download_image(png_url)
            if exists:
                has_new_card = True
                invalid_counts[prefix] = 0
                star = '⭐5' if prefix[1] == '5' else '⭐6' if prefix[1] == '6' else '⭐7' if prefix[1] == '7' else '⭐8'

                print(f"发现新卡面，调用 PaddleOCR: {jpg_url}")
                en_tag, jp_text = paddle_ocr(jpg_url)
                en_tag = fuzzy_match_tag(en_tag, TOPIC_MAP)

                caption = f"#{en_tag}\n\n{star}\n\n{png_url}"
                if len(caption) > 1000: caption = caption[:1000] + "..."

                target_thread_id = TOPIC_MAP.get(en_tag, DEFAULT_TOPIC_ID)
                
                # 1. 先推送 PNG (带上完整的 Tag、星级和说明)
                send_photo_to_telegram(png_url, caption, thread_id=target_thread_id)
                
                # 停顿 1-2 秒，防止连续发两条消息太快被 Telegram 接口限流报错
                time.sleep(1)
                
                # 2. 紧接着推送 JPG 作为一条独立的新消息
                send_photo_to_telegram(jpg_url, caption, thread_id=target_thread_id)
                
                start_points[prefix] += 1

                # === 每次找到新卡后保存状态 ===
                save_state(start_points) 
            else:
                invalid_counts[prefix] += 1
                start_points[prefix] += 1
                # 即使没找到也要保存跳过的进度
                save_state(start_points)

            # 无效连续判定
            if invalid_counts[prefix] >= max_invalid:
                completed_categories.add(prefix)
                
                # === 核心修复逻辑：回退多走的探测步数 ===
                # 既然连续 20 个都没有，说明我们撞到了目前的“更新尽头”。
                # 我们要把序号往回倒退 20，确保下次运行时，能从第一个没找到的号码重新开始测。
                start_points[prefix] -= max_invalid
                save_state(start_points) 
                print(f"[{prefix}] 连续 {max_invalid} 次无效，已回退序号至 {start_points[prefix]}，等待下次运行。")
            time.sleep(1) # Github Actions 速度很快，休眠一小会儿防止被拉黑

        if len(completed_categories) >= len(start_points):
            print("所有分类遍历完毕，任务结束。")
            break

def main():
    print(f"{datetime.now()} - 乃木恋 Bot 启动，使用 PaddleOCR 引擎...")
    download_task()

if __name__ == "__main__":
    main()
