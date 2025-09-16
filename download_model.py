import os
import sys
import requests
from tqdm import tqdm

# 检查命令行参数数量是否正确
if len(sys.argv) != 2:
    print('你必须输入模型名称作为参数，例如：download_model.py 124M')
    sys.exit(1)

# 获取从命令行传入的模型名称
model = sys.argv[1]

# 定义模型文件存储目录
subdir = os.path.join('models', model)
# 如果目录不存在则创建
if not os.path.exists(subdir):
    os.makedirs(subdir)
# 处理Windows系统的路径斜杠问题
subdir = subdir.replace('\\', '/')

# 需要下载的模型相关文件列表
for filename in ['checkpoint', 'encoder.json', 'hparams.json', 
                 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 
                 'model.ckpt.meta', 'vocab.bpe']:

    # 构建文件下载URL
    url = f"https://openaipublic.blob.core.windows.net/gpt-2/{subdir}/{filename}"
    # 发送GET请求，流式获取数据
    response = requests.get(url, stream=True)

    # 保存文件到本地
    with open(os.path.join(subdir, filename), 'wb') as file:
        # 获取文件总大小
        file_size = int(response.headers["content-length"])
        # 分块大小（1000字节），接近以太网数据包大小（约1500字节）
        chunk_size = 1000
        # 创建进度条显示下载进度
        with tqdm(ncols=100, desc=f"正在下载 {filename}", total=file_size, unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)  # 写入文件块
                pbar.update(chunk_size)  # 更新进度条