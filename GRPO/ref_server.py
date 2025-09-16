from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import io
import queue
import threading
from bottle import Bottle, request
import os

# 环境配置
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# 工具函数：张量与字节流转换
def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(b), weights_only=True)

def make_bytes_list(blist: list[bytes]) -> bytes:
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))  # 写入列表长度
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))  # 写入每个元素长度
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b: bytes) -> list[bytes]:
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')  # 读取列表长度
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')  # 读取元素长度
        blist.append(buffer.read(l))
    return blist

# 参考模型初始化
model_path = "/data2/Qwen/Qwen2.5-7B"
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="sdpa"
).to('cuda')
ref_model.eval()
ref_model.requires_grad_(False)

# 计算每个token的log概率
def get_per_token_logps(input_ids: torch.Tensor) -> torch.Tensor:
    logits = ref_model(input_ids).logits  # (batch_size, seq_len, vocab_size)
    logits = logits[:, :-1, :]  # 去掉最后一个logit（无对应输入token）
    input_ids = input_ids[:, 1:]  # 去掉第一个输入token（无对应logit）
    
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)  # 转为对数概率
        # 取出每个输入token对应的概率
        token_logp = torch.gather(log_probs, 1, ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_logp)
    return torch.stack(per_token_logps)

# 队列用于存储数据（生产者-消费者模式）
raw_queue = queue.LifoQueue()
result_queue = queue.LifoQueue()

# 启动HTTP服务
app = Bottle()

@app.route('/upload', method='POST')
def upload_data():
    """接收训练端上传的样本数据"""
    data_bytes = request.body.read()
    data_list = bytes_list_to_list(data_bytes)
    if len(data_list) not in (3, 4):
        return b'error'
    
    # 解析数据
    base_info = json.loads(data_list[0])
    parsed_data = {
        'base': base_info,
        'inputs': bytes_to_tensor(data_list[1]),
        'rewards': bytes_to_tensor(data_list[2])
    }
    if len(data_list) == 4:
        parsed_data['gen_logps'] = bytes_to_tensor(data_list[3])
    
    raw_queue.put(parsed_data)
    print(f"接收数据: 输入形状{parsed_data['inputs'].shape}, 奖励{parsed_data['rewards']}")
    return b'success'

@app.route('/get', method='GET')
def get_data():
    """向训练端返回带参考logits的数据"""
    if result_queue.empty():
        return b'empty'
    return result_queue.get()

def run_server():
    """启动服务端（ tornado 引擎更高效）"""
    bottle.run(app, host='0.0.0.0', port=59875, server='tornado')

def process_data():
    """后台处理数据，计算参考logits"""
    while True:
        data = raw_queue.get()  # 阻塞等待数据
        prompt_length = data['base']['plen']
        
        # 计算参考模型的token logps
        with torch.inference_mode():
            ref_logps = get_per_token_logps(data['inputs'].to(ref_model.device))
        ref_logps = ref_logps[:, prompt_length-1:]  # 只保留生成部分的logps
        
        # 打包返回数据
        return_list = [
            json.dumps(data['base']).encode(),
            tensor_to_bytes(data['inputs']),
            tensor_to_bytes(data['rewards']),
            tensor_to_bytes(ref_logps)
        ]
        if 'gen_logps' in data:
            return_list.append(tensor_to_bytes(data['gen_logps']))
        
        result_queue.put(make_bytes_list(return_list))

if __name__ == '__main__':
    import json
    # 启动服务端和数据处理线程
    threading.Thread(target=run_server, daemon=True).start()
    threading.Thread(target=process_data, daemon=True).start()
    print("参考模型服务端已启动，监听端口59875...")
    while True:
        time.sleep(3600)  # 保持进程运行