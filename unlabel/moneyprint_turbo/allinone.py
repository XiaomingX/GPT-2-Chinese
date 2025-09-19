import os
import json
import shutil
import toml
import requests
import argparse
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel

class Config:
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.config_file = os.path.join(self.root_dir, "config.toml")
        self.example_file = os.path.join(self.root_dir, "config.example.toml")
        self._load_config()
        
    def _load_config(self):
        # 确保配置文件存在
        if os.path.isdir(self.config_file):
            shutil.rmtree(self.config_file)
            
        if not os.path.isfile(self.config_file) and os.path.isfile(self.example_file):
            shutil.copyfile(self.example_file, self.config_file)
            logger.info("已从示例文件复制配置")
            
        # 加载配置
        try:
            with open(self.config_file, "r", encoding="utf-8-sig") as f:
                self.data = toml.load(f)
        except Exception as e:
            logger.warning(f"加载配置失败，使用默认配置: {e}")
            self.data = {}
            
        # 提取配置项
        self.app = self.data.get("app", {})
        self.whisper = self.data.get("whisper", {})
        self.proxy = self.data.get("proxy", {})

# 全局配置
config = Config()

# LLM调用 - 简化版，只保留核心提供商
def generate_response(prompt: str, max_retries=3) -> str:
    llm_provider = config.app.get("llm_provider", "openai")
    
    for i in range(max_retries):
        try:
            if llm_provider == "openai":
                return _openai_response(prompt)
            elif llm_provider == "g4f":
                return _g4f_response(prompt)
            elif llm_provider == "azure":
                return _azure_response(prompt)
            else:
                raise ValueError(f"不支持的LLM提供商: {llm_provider}")
        except Exception as e:
            logger.warning(f"生成响应失败 (尝试 {i+1}/{max_retries}): {e}")
            if i == max_retries - 1:
                return f"错误: {str(e)}"

def _openai_response(prompt: str) -> str:
    from openai import OpenAI
    
    client = OpenAI(
        api_key=config.app.get("openai_api_key"),
        base_url=config.app.get("openai_base_url", "https://api.openai.com/v1")
    )
    
    response = client.chat.completions.create(
        model=config.app.get("openai_model_name", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def _g4f_response(prompt: str) -> str:
    import g4f
    
    response = g4f.ChatCompletion.create(
        model=config.app.get("g4f_model_name", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": prompt}]
    )
    return response

def _azure_response(prompt: str) -> str:
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        api_key=config.app.get("azure_api_key"),
        api_version=config.app.get("azure_api_version", "2024-02-15-preview"),
        azure_endpoint=config.app.get("azure_base_url")
    )
    
    response = client.chat.completions.create(
        model=config.app.get("azure_model_name"),
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 视频脚本生成
def generate_script(video_subject: str, language: str = "", paragraph_number: int = 1) -> str:
    prompt = f"""
# 视频脚本生成器
根据主题生成视频脚本，要求：
1. 生成{paragraph_number}个段落
2. 直接切入主题，不要多余的开场白
3. 不要任何格式标记或标题
4. 不要包含"画外音"、"旁白"等提示词
5. 用与主题相同的语言回应

视频主题: {video_subject}
    """.strip()
    
    if language:
        prompt += f"\n语言: {language}"

    for i in range(3):  # 最多重试3次
        try:
            response = generate_response(prompt)
            if response and "错误: " not in response:
                # 清理响应内容
                response = response.replace("*", "").replace("#", "")
                response = response.replace("\n\n", "\n").strip()
                return response
        except Exception as e:
            logger.error(f"生成脚本失败: {e}")
    
    return "无法生成脚本，请检查配置后重试"

# 视频搜索词生成
def generate_search_terms(video_subject: str, video_script: str, amount: int = 5) -> list:
    prompt = f"""
生成{amount}个英文视频搜索词，需满足：
1. 以JSON数组形式返回，仅包含搜索词
2. 每个搜索词1-3个单词，包含视频主题
3. 与视频主题和脚本内容相关

视频主题: {video_subject}
视频脚本: {video_script}

输出示例: ["term1", "term2", "term3", "term4", "term5"]
    """.strip()

    for i in range(3):  # 最多重试3次
        try:
            response = generate_response(prompt)
            if response and "错误: " not in response:
                # 提取JSON数组
                terms = json.loads(response)
                if isinstance(terms, list) and len(terms) >= amount:
                    return terms[:amount]
        except Exception as e:
            logger.error(f"生成搜索词失败: {e}")
    
    # 如果生成失败，返回基于主题的简单搜索词
    return [video_subject] * amount

# 视频下载功能
def download_video(video_url: str, save_dir: str = "downloads") -> str:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成文件名
    filename = f"video_{hash(video_url)}.mp4"
    video_path = os.path.join(save_dir, filename)
    
    # 检查文件是否已存在
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        logger.info(f"视频已存在: {video_path}")
        return video_path
    
    # 下载视频
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        
        response = requests.get(
            video_url,
            headers=headers,
            proxies=config.proxy,
            verify=False,
            timeout=60
        )
        
        with open(video_path, "wb") as f:
            f.write(response.content)
        
        # 验证视频文件
        with VideoFileClip(video_path) as clip:
            if clip.duration > 0:
                logger.info(f"视频下载成功: {video_path}")
                return video_path
        
        # 如果验证失败，删除文件
        os.remove(video_path)
        return ""
        
    except Exception as e:
        logger.error(f"视频下载失败: {e}")
        if os.path.exists(video_path):
            os.remove(video_path)
        return ""

# 搜索并下载视频
def search_and_download_videos(search_terms: list, max_videos: int = 3, source: str = "pexels") -> list:
    video_paths = []
    
    for term in search_terms:
        if len(video_paths) >= max_videos:
            break
            
        logger.info(f"搜索视频: {term}")
        videos = []
        
        if source == "pexels":
            videos = _search_pexels(term)
        elif source == "pixabay":
            videos = _search_pixabay(term)
        
        # 下载找到的视频
        for video in videos:
            if len(video_paths) >= max_videos:
                break
                
            path = download_video(video["url"])
            if path:
                video_paths.append(path)
    
    return video_paths

def _search_pexels(search_term: str) -> list:
    try:
        api_key = config.app.get("pexels_api_key")
        if not api_key:
            logger.warning("未配置Pexels API密钥")
            return []
            
        url = f"https://api.pexels.com/videos/search?query={search_term}&per_page=5"
        headers = {"Authorization": api_key}
        
        response = requests.get(
            url,
            headers=headers,
            proxies=config.proxy,
            verify=False,
            timeout=30
        )
        
        data = response.json()
        videos = []
        
        for item in data.get("videos", []):
            # 选择最佳质量的视频
            for video_file in item.get("video_files", []):
                if video_file.get("width") >= 1080:
                    videos.append({
                        "url": video_file.get("link"),
                        "duration": item.get("duration")
                    })
                    break
                    
        return videos
    except Exception as e:
        logger.error(f"Pexels搜索失败: {e}")
        return []

def _search_pixabay(search_term: str) -> list:
    try:
        api_key = config.app.get("pixabay_api_key")
        if not api_key:
            logger.warning("未配置Pixabay API密钥")
            return []
            
        url = f"https://pixabay.com/api/videos/?q={search_term}&per_page=5&key={api_key}"
        
        response = requests.get(
            url,
            proxies=config.proxy,
            verify=False,
            timeout=30
        )
        
        data = response.json()
        videos = []
        
        for item in data.get("hits", []):
            # 选择最佳质量的视频
            video_files = item.get("videos", {})
            best_video = video_files.get("large", video_files.get("medium"))
            
            if best_video:
                videos.append({
                    "url": best_video.get("url"),
                    "duration": item.get("duration")
                })
                    
        return videos
    except Exception as e:
        logger.error(f"Pixabay搜索失败: {e}")
        return []

# 语音转文字 (Whisper)
def transcribe_audio(audio_path: str) -> str:
    try:
        model_size = config.whisper.get("model_size", "base")
        device = config.whisper.get("device", "cpu")
        compute_type = config.whisper.get("compute_type", "int8")
        
        logger.info(f"加载Whisper模型: {model_size}")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        logger.info(f"开始转录音频: {audio_path}")
        segments, info = model.transcribe(audio_path)
        
        logger.info(f"检测到语言: {info.language}")
        transcription = " ".join([segment.text for segment in segments])
        
        return transcription
    except Exception as e:
        logger.error(f"音频转录失败: {e}")
        return ""

# 主流程
def main():
    parser = argparse.ArgumentParser(description="视频内容生成工具")
    parser.add_argument("--subject", required=True, help="视频主题")
    parser.add_argument("--lang", default="zh-CN", help="语言")
    parser.add_argument("--paragraphs", type=int, default=1, help="段落数量")
    parser.add_argument("--terms", type=int, default=5, help="搜索词数量")
    parser.add_argument("--videos", type=int, default=3, help="下载视频数量")
    parser.add_argument("--source", default="pexels", help="视频来源 (pexels/pixabay)")
    args = parser.parse_args()
    
    logger.info("===== 开始视频内容生成流程 =====")
    
    # 1. 生成视频脚本
    logger.info("\n===== 生成视频脚本 =====")
    script = generate_script(
        video_subject=args.subject,
        language=args.lang,
        paragraph_number=args.paragraphs
    )
    print(f"生成的脚本:\n{script}\n")
    
    # 2. 生成搜索词
    logger.info("\n===== 生成视频搜索词 =====")
    search_terms = generate_search_terms(
        video_subject=args.subject,
        video_script=script,
        amount=args.terms
    )
    print(f"生成的搜索词:\n{search_terms}\n")
    
    # 3. 搜索并下载视频
    logger.info("\n===== 搜索并下载视频 =====")
    video_paths = search_and_download_videos(
        search_terms=search_terms,
        max_videos=args.videos,
        source=args.source
    )
    print(f"下载的视频路径:\n{video_paths}\n")
    
    # 4. 示例：如果有音频文件，进行转录
    if video_paths:
        logger.info("\n===== 示例：从视频中提取音频并转录 =====")
        try:
            # 从第一个视频中提取音频
            with VideoFileClip(video_paths[0]) as video:
                audio_path = "temp_audio.wav"
                video.audio.write_audiofile(audio_path)
                
                # 转录音频
                transcription = transcribe_audio(audio_path)
                print(f"音频转录结果:\n{transcription}\n")
                
                # 清理临时文件
                os.remove(audio_path)
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
    
    logger.info("\n===== 流程完成 =====")

if __name__ == "__main__":
    main()