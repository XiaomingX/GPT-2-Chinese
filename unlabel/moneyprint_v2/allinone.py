import os
import sys
import json
import random
import zipfile
import requests
import platform
import schedule
import subprocess
from uuid import uuid4
from termcolor import colored
from prettytable import PrettyTable
import g4f
import srt_equalizer

# 配置与路径设置
ROOT_DIR = os.path.dirname(sys.path[0])
CACHE_DIR = os.path.join(ROOT_DIR, '.mp')
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.json')
SONGS_DIR = os.path.join(ROOT_DIR, 'Songs')
BANNER_PATH = os.path.join(ROOT_DIR, 'assets', 'banner.txt')

# 常量定义
OPTIONS = [
    "YouTube Shorts Automation",
    "Twitter Bot",
    "Affiliate Marketing",
    "Outreach",
    "Quit"
]

TWITTER_OPTIONS = ["Post something", "Show all Posts", "Setup CRON Job", "Quit"]
TWITTER_CRON_OPTIONS = ["Once a day", "Twice a day", "Thrice a day", "Quit"]
YOUTUBE_OPTIONS = ["Upload Short", "Show all Shorts", "Setup CRON Job", "Quit"]
YOUTUBE_CRON_OPTIONS = ["Once a day", "Twice a day", "Thrice a day", "Quit"]

# 配置读取函数（统一处理所有配置项）
def get_config(key, default=None):
    if not os.path.exists(CONFIG_PATH):
        return default
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config.get(key, default)

# 缓存文件路径生成
def get_cache_file(provider):
    files = {
        'twitter': 'twitter.json',
        'youtube': 'youtube.json',
        'afm': 'afm.json',
        'results': 'scraper_results.csv'
    }
    return os.path.join(CACHE_DIR, files.get(provider, ''))

# 初始化文件夹结构
def init_folders():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(colored(f"创建缓存文件夹: {CACHE_DIR}", "green"))
    
    # 初始化缓存文件
    for provider in ['twitter', 'youtube', 'afm']:
        path = get_cache_file(provider)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump({"accounts" if provider != 'afm' else "products": []}, f, indent=4)

    # 初始化歌曲文件夹
    if not os.path.exists(SONGS_DIR):
        os.makedirs(SONGS_DIR)
        fetch_songs()

# 下载歌曲
def fetch_songs():
    try:
        print(colored("获取背景音乐...", "blue"))
        zip_url = get_config('zip_url', "https://filebin.net/bb9ewdtckolsf3sg/drive-download-20240209T180019Z-001.zip")
        response = requests.get(zip_url)
        
        zip_path = os.path.join(SONGS_DIR, 'songs.zip')
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(SONGS_DIR)
        
        os.remove(zip_path)
        print(colored("背景音乐下载完成", "green"))
    except Exception as e:
        print(colored(f"音乐下载失败: {str(e)}", "red"))

# 账户管理
def get_accounts(provider):
    path = get_cache_file(provider)
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('accounts', [])

def add_account(provider, account_data):
    accounts = get_accounts(provider)
    accounts.append(account_data)
    with open(get_cache_file(provider), 'w') as f:
        json.dump({'accounts': accounts}, f, indent=4)

def remove_account(provider, account_id):
    accounts = get_accounts(provider)
    accounts = [acc for acc in accounts if acc['id'] != account_id]
    with open(get_cache_file(provider), 'w') as f:
        json.dump({'accounts': accounts}, f, indent=4)

# 产品管理（联盟营销）
def get_products():
    path = get_cache_file('afm')
    with open(path, 'r') as f:
        return json.load(f).get('products', [])

def add_product(product_data):
    products = get_products()
    products.append(product_data)
    with open(get_cache_file('afm'), 'w') as f:
        json.dump({'products': products}, f, indent=4)

# 模型解析
def get_model(model_name):
    models = {
        "gpt4": g4f.models.gpt_4,
        "gpt35_turbo": g4f.models.gpt_4o_mini,
        "llama2_7b": g4f.models.llama2_7b,
        "llama2_13b": g4f.models.llama2_13b,
        "llama2_70b": g4f.models.llama2_70b,
        "mixtral_8x7b": g4f.models.mixtral_8x7b
    }
    return models.get(model_name, g4f.models.gpt_4o_mini)

# 辅助函数
def print_banner():
    if os.path.exists(BANNER_PATH):
        with open(BANNER_PATH, 'r') as f:
            print(colored(f.read(), "green"))

def choose_random_song():
    songs = os.listdir(SONGS_DIR)
    return os.path.join(SONGS_DIR, random.choice(songs)) if songs else None

def close_browsers():
    try:
        print(colored("关闭浏览器进程...", "blue"))
        if platform.system() == "Windows":
            os.system("taskkill /f /im firefox.exe")
        else:
            os.system("pkill firefox")
    except Exception as e:
        print(colored(f"关闭浏览器失败: {str(e)}", "red"))

def clear_temp_files():
    for file in os.listdir(CACHE_DIR):
        if not file.endswith('.json'):
            os.remove(os.path.join(CACHE_DIR, file))

# 平台功能类（简化版）
class Twitter:
    def __init__(self, account_id, nickname, profile, topic):
        self.account_id = account_id
        self.nickname = nickname
        self.profile = profile
        self.topic = topic

    def post(self):
        print(colored(f"[{self.nickname}] 发布推文: 主题={self.topic}", "green"))
        # 实际发布逻辑可在此处添加

    def get_posts(self):
        return [{"date": "2024-05-01", "content": f"测试推文: {self.topic}"}]

class YouTube:
    def __init__(self, account_id, nickname, profile, niche, language):
        self.account_id = account_id
        self.nickname = nickname
        self.profile = profile
        self.niche = niche
        self.language = language

    def generate_video(self):
        print(colored(f"[{self.nickname}] 生成视频: 领域={self.niche}", "green"))
        song = choose_random_song()
        if song:
            print(colored(f"使用背景音乐: {os.path.basename(song)}", "blue"))

    def upload_video(self):
        print(colored(f"[{self.nickname}] 上传视频到YouTube", "green"))

    def get_videos(self):
        return [{"date": "2024-05-01", "title": f"测试视频: {self.niche}"}]

class AffiliateMarketing:
    def __init__(self, link, profile, account_id, nickname, topic):
        self.link = link
        self.profile = profile
        self.account_id = account_id
        self.nickname = nickname
        self.topic = topic

    def generate_pitch(self):
        print(colored(f"生成推广文案: {self.link}", "green"))

    def share_pitch(self, platform):
        print(colored(f"通过{platform}分享推广: {self.link}", "green"))

class Outreach:
    def start(self):
        print(colored("启动 outreach 流程", "green"))

# 主菜单逻辑
def main_menu():
    while True:
        print("\n" + "="*30)
        print(colored("主菜单", "cyan"))
        for i, opt in enumerate(OPTIONS, 1):
            print(colored(f" {i}. {opt}", "cyan"))
        print("="*30)

        try:
            choice = int(input("请选择功能: ").strip())
            if choice == 1:
                youtube_menu()
            elif choice == 2:
                twitter_menu()
            elif choice == 3:
                afm_menu()
            elif choice == 4:
                Outreach().start()
            elif choice == 5:
                print(colored("程序退出", "blue"))
                sys.exit(0)
            else:
                print(colored("无效选项", "red"))
        except ValueError:
            print(colored("请输入数字", "red"))

# YouTube子菜单
def youtube_menu():
    accounts = get_accounts('youtube')
    if not accounts:
        print(colored("没有YouTube账户，创建一个新账户", "yellow"))
        new_account = {
            "id": str(uuid4()),
            "nickname": input("账户昵称: "),
            "firefox_profile": input("Firefox配置路径: "),
            "niche": input("内容领域: "),
            "language": input("语言: "),
            "use_g4f": input("使用G4F生成图片? (1/2): ") == "1",
            "videos": []
        }
        if not new_account["use_g4f"]:
            new_account["worker_url"] = input("Cloudflare Worker URL: ")
        add_account('youtube', new_account)
        accounts = get_accounts('youtube')

    # 显示账户列表
    table = PrettyTable(["ID", "昵称", "领域"])
    for i, acc in enumerate(accounts, 1):
        table.add_row([i, acc["nickname"], acc["niche"]])
    print(table)

    try:
        sel = int(input("选择账户: ")) - 1
        account = accounts[sel]
        yt = YouTube(**account)
        
        while True:
            print("\n" + "="*30)
            print(colored("YouTube功能", "cyan"))
            for i, opt in enumerate(YOUTUBE_OPTIONS, 1):
                print(colored(f" {i}. {opt}", "cyan"))
            print("="*30)

            opt = int(input("选择操作: ").strip())
            if opt == 1:
                yt.generate_video()
                if input("上传视频? (y/n): ").lower() == 'y':
                    yt.upload_video()
            elif opt == 2:
                videos = yt.get_videos()
                table = PrettyTable(["ID", "日期", "标题"])
                for i, v in enumerate(videos, 1):
                    table.add_row([i, v["date"], v["title"]])
                print(table)
            elif opt == 3:
                setup_cron('youtube', account["id"])
            elif opt == 4:
                break
    except (IndexError, ValueError):
        print(colored("无效选择", "red"))

# Twitter子菜单
def twitter_menu():
    accounts = get_accounts('twitter')
    if not accounts:
        print(colored("没有Twitter账户，创建一个新账户", "yellow"))
        new_account = {
            "id": str(uuid4()),
            "nickname": input("账户昵称: "),
            "firefox_profile": input("Firefox配置路径: "),
            "topic": input("内容主题: "),
            "posts": []
        }
        add_account('twitter', new_account)
        accounts = get_accounts('twitter')

    # 显示账户列表
    table = PrettyTable(["ID", "昵称", "主题"])
    for i, acc in enumerate(accounts, 1):
        table.add_row([i, acc["nickname"], acc["topic"]])
    print(table)

    try:
        sel = int(input("选择账户: ")) - 1
        account = accounts[sel]
        tw = Twitter(** account)
        
        while True:
            print("\n" + "="*30)
            print(colored("Twitter功能", "cyan"))
            for i, opt in enumerate(TWITTER_OPTIONS, 1):
                print(colored(f" {i}. {opt}", "cyan"))
            print("="*30)

            opt = int(input("选择操作: ").strip())
            if opt == 1:
                tw.post()
            elif opt == 2:
                posts = tw.get_posts()
                table = PrettyTable(["ID", "日期", "内容"])
                for i, p in enumerate(posts, 1):
                    table.add_row([i, p["date"], p["content"]])
                print(table)
            elif opt == 3:
                setup_cron('twitter', account["id"])
            elif opt == 4:
                break
    except (IndexError, ValueError):
        print(colored("无效选择", "red"))

# 联盟营销子菜单
def afm_menu():
    products = get_products()
    if not products:
        print(colored("没有产品，添加一个新推广产品", "yellow"))
        aff_link = input("推广链接: ")
        tw_id = input("关联Twitter账户ID: ")
        add_product({
            "id": str(uuid4()),
            "affiliate_link": aff_link,
            "twitter_uuid": tw_id
        })
        products = get_products()

    # 显示产品列表
    table = PrettyTable(["ID", "推广链接", "关联Twitter ID"])
    for i, p in enumerate(products, 1):
        table.add_row([i, p["affiliate_link"], p["twitter_uuid"]])
    print(table)

    try:
        sel = int(input("选择产品: ")) - 1
        product = products[sel]
        # 查找关联的Twitter账户
        tw_accounts = get_accounts('twitter')
        account = next((a for a in tw_accounts if a["id"] == product["twitter_uuid"]), None)
        if account:
            afm = AffiliateMarketing(product["affiliate_link"], account["firefox_profile"], 
                                    account["id"], account["nickname"], account["topic"])
            afm.generate_pitch()
            afm.share_pitch("twitter")
        else:
            print(colored("关联的Twitter账户不存在", "red"))
    except (IndexError, ValueError):
        print(colored("无效选择", "red"))

# 定时任务设置
def setup_cron(provider, account_id):
    cron_script = os.path.join(ROOT_DIR, 'src', 'cron.py')
    cmd = f"python {cron_script} {provider} {account_id}"
    
    def job():
        subprocess.run(cmd, shell=True)

    print("\n" + "="*30)
    print(colored("定时设置", "cyan"))
    options = TWITTER_CRON_OPTIONS if provider == 'twitter' else YOUTUBE_CRON_OPTIONS
    for i, opt in enumerate(options, 1):
        print(colored(f" {i}. {opt}", "cyan"))
    print("="*30)

    try:
        opt = int(input("选择频率: ").strip())
        if opt == 1:
            schedule.every(1).day.do(job)
        elif opt == 2:
            schedule.every().day.at("10:00").do(job)
            schedule.every().day.at("16:00").do(job)
        elif opt == 3:
            schedule.every().day.at("08:00").do(job)
            schedule.every().day.at("12:00").do(job)
            schedule.every().day.at("18:00").do(job)
        print(colored("定时任务设置完成", "green"))
    except ValueError:
        print(colored("无效选择", "red"))

# 程序入口
if __name__ == "__main__":
    print_banner()
    init_folders()
    clear_temp_files()
    close_browsers()
    main_menu()