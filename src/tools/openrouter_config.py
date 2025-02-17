import os
import time
import logging
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from typing import Optional, Dict, Any
import requests

# 设置日志记录
logger = logging.getLogger('api_calls')
logger.setLevel(logging.DEBUG)

# 移除所有现有的处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 设置文件处理器
log_file = os.path.join(log_dir, f'api_calls_{time.strftime("%Y%m%d")}.log')
print(f"Creating log file at: {log_file}")

try:
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.DEBUG)
    print("Successfully created file handler")
except Exception as e:
    print(f"Error creating file handler: {str(e)}")

# 设置控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 立即测试日志记录
logger.debug("Logger initialization completed")
logger.info("API logging system started")

# 状态图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "⟳"


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 验证环境变量
api_key = os.getenv("OPENROUTER_API_KEY")
default_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 OPENROUTER_API_KEY 环境变量")
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

logger.info(f"{SUCCESS_ICON} OpenRouter 配置初始化成功")

def call_openrouter_api(messages, model=None, temperature=0.7):
    """调用 OpenRouter API 的基础函数"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model or default_model,
        "messages": messages,
        "temperature": temperature
    }

    try:
        logger.info(f"{WAIT_ICON} 正在调用 OpenRouter API...")
        logger.info(f"请求内容: {str(messages)[:500]}..." if len(
            str(messages)) > 500 else f"请求内容: {messages}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"{SUCCESS_ICON} API 调用成功")
            return data
        else:
            logger.error(f"{ERROR_ICON} API 调用失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"{ERROR_ICON} API 调用异常: {str(e)}")
        raise e

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=300
)
def get_chat_completion(messages, model=None, temperature=0.7):
    """获取聊天完成结果，包含重试逻辑"""
    try:
        response = call_openrouter_api(
            messages=messages,
            model=model,
            temperature=temperature
        )
        
        if response is None:
            logger.error(f"{ERROR_ICON} API 返回空响应")
            return None

        content = response["choices"][0]["message"]["content"]
        logger.info(f"{SUCCESS_ICON} 成功获取响应")
        logger.debug(f"响应内容: {content[:500]}..." if len(
            content) > 500 else f"响应内容: {content}")
        
        return content

    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None
