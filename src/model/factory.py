from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
import yaml
import os
from utils.log_utils import app_logger
from utils.file_utils import log_event

# 加载 .env 文件（优先级：先尝试项目根目录，再尝试 src 父目录）
def _load_dotenv():
    """尝试加载 .env 文件，失败时静默跳过"""
    try:
        from dotenv import load_dotenv as _load
        # 尝试多个可能的 .env 路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            os.path.join(current_dir, '..', '..', '.env'),      # 项目根目录
            os.path.join(current_dir, '..', '.env'),             # src 目录
            os.path.join(os.getcwd(), '.env'),                   # 当前工作目录
        ]
        for env_path in search_paths:
            if os.path.exists(env_path):
                _load(env_path)
                app_logger.info(f'[DOTENV] 已加载环境变量文件: {env_path}')
                return
        app_logger.info('[DOTENV] 未找到 .env 文件，使用系统环境变量')
    except ImportError:
        app_logger.warning('[DOTENV] python-dotenv 未安装，跳过 .env 文件加载')
    except Exception as e:
        app_logger.warning(f'[DOTENV] 加载 .env 失败: {str(e)}')

_load_dotenv()


def _override_from_env(config: dict) -> dict:
    """用环境变量覆盖配置中的敏感信息，确保密钥不会硬编码在 YAML 中"""
    env_mappings = {
        'QWEN_API_KEY': ('qwen_api_key', None),                         # 顶层 key
        'TAVILY_API_KEY': ('api_key', 'tavily'),                         # config['tavily']['api_key']
        'PEXELS_API_KEY': ('api_key', 'pexels'),                         # config['pexels']['api_key']
        'TENCENTCLOUD_SECRET_ID': ('secret_id', 'tencentcloud'),         # config['tencentcloud']['secret_id']
        'TENCENTCLOUD_SECRET_KEY': ('secret_key', 'tencentcloud'),       # config['tencentcloud']['secret_key']
    }

    for env_var, (config_key, parent_key) in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            if parent_key:
                # 确保父级是一个 dict（YAML 中空 section 可能被解析为 None）
                if not isinstance(config.get(parent_key), dict):
                    config[parent_key] = {}
                config[parent_key][config_key] = env_value
            else:
                config[config_key] = env_value

    return config


def load_config():
    """加载配置文件，支持类似 Spring Boot 的多环境 Profile 切换。
    敏感信息（API Key / Secret）优先从环境变量读取，其次从 YAML 配置文件读取。
    """
    import os
    import yaml
    from utils.log_utils import app_logger

    # 获取当前运行环境，支持 SPRING_PROFILES_ACTIVE 或 APP_PROFILES_ACTIVE 环境变量，默认 dev
    profile = os.getenv('SPRING_PROFILES_ACTIVE', os.getenv('APP_PROFILES_ACTIVE', 'dev'))

    def load_yaml_file(filename):
        path = os.path.join(os.path.dirname(__file__), '..', 'config', filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                app_logger.error(f'[CONFIG_LOAD] 解析 {filename} 失败: {str(e)}')
        return {}

    # 1. 加载主配置文件 (作为默认基础配置)
    config = load_yaml_file('rag.yml')

    # 2. 如果指定了非 dev 环境，加载对应的环境配置文件并覆盖基础配置
    if profile != 'dev':
        env_filename = f'rag-{profile}.yml'
        env_config = load_yaml_file(env_filename)
        if env_config:
            app_logger.info(f'[CONFIG_LOAD] 检测到 Profile: {profile}，已合并环境配置: {env_filename}')
            config.update(env_config)
        else:
            app_logger.warning(f'[CONFIG_LOAD] 指定了 Profile: {profile}，但未找到配置文件: {env_filename}')

    # 3. 加载工具配置文件
    tool_config = load_yaml_file('tool.yml')
    if tool_config:
        config.update(tool_config)

    # 4. 加载并发配置文件
    concurrency_config = load_yaml_file('concurrency.yml')
    if concurrency_config:
        config.update(concurrency_config)
    else:
        # 设置默认并发配置
        if 'concurrency' not in config:
            config['concurrency'] = {
                'thread_pool': {'max_workers': 20},
                'request_processing': {'max_concurrent_requests': 50, 'timeout_seconds': 300},
                'agent': {'max_concurrent_agents': 10, 'agent_timeout_seconds': 120}
            }

    # 5. 用环境变量覆盖敏感配置（优先级最高）
    config = _override_from_env(config)

    return config

def load_qdrant_config():
    """加载 Qdrant 配置文件，支持多环境 Profile 切换"""
    import os
    import yaml
    from utils.log_utils import app_logger

    profile = os.getenv('SPRING_PROFILES_ACTIVE', os.getenv('APP_PROFILES_ACTIVE', 'dev'))

    def load_yaml_file(filename):
        path = os.path.join(os.path.dirname(__file__), '..', 'config', filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                app_logger.error(f'[CONFIG_LOAD] 解析 {filename} 失败: {str(e)}')
        return {}

    config = load_yaml_file('qdrant.yml')

    if profile != 'dev':
        env_filename = f'qdrant-{profile}.yml'
        env_config = load_yaml_file(env_filename)
        if env_config:
            app_logger.info(f'[CONFIG_LOAD] Qdrant 检测到 Profile: {profile}，已合并环境配置: {env_filename}')
            config.update(env_config)
        else:
            app_logger.warning(f'[CONFIG_LOAD] Qdrant 指定了 Profile: {profile}，但未找到配置文件: {env_filename}')

    return config

def create_chat_model(model_name_param: str = None):
    """创建聊天模型 (使用 OpenAI 兼容协议连接 DashScope)"""
    config = load_config()
    model_name = model_name_param if model_name_param else config.get('chat_model_name', 'qwen3.5-flash')
    temperature = config.get('qwen_temperature', 0.1) # 默认使用低采样率
    max_tokens = config.get('qwen_max_tokens', 2048)
    
    app_logger.info('[MODEL_CREATION] 使用 ChatOpenAI 桥接 DashScope | model: ' + model_name)
    
    # 构建模型参数
    # 注意：LangChain 的 agent.stream() 无法解析 reasoning_content 字段（DashScope 思考模式专属），
    # 若启用 enable_thinking，模型将把回答放入 reasoning_content 而非 content，导致 output 为空。
    # 因此在 LangChain Agent 场景下必须禁用 enable_thinking，保持标准 content 输出。
    model_kwargs = {
        'model': model_name,
        'openai_api_key': config['qwen_api_key'],
        'openai_api_base': "https://dashscope.aliyuncs.com/compatible-mode/v1",
        'temperature': temperature,
        'max_tokens': max_tokens,
        'streaming': True,
        'extra_body': {'enable_thinking': False}
    }
    
    app_logger.info('[MODEL_CREATION] LangChain Agent 模式，已禁用 enable_thinking（防止 reasoning_content 导致 output 为空）')
    
    # 使用 OpenAI 兼容模式地址，这通常比原来的 ChatTongyi 驱动处理工具调用时更稳定
    model = ChatOpenAI(**model_kwargs)
    
    app_logger.info('[MODEL_CREATED] 兼容模式聊天模型创建成功 | model_name: ' + model_name)
    return model

def create_embedding_model():
    """创建嵌入模型"""
    config = load_config()
    model_name = config['embedding_model_name']
    app_logger.info('[EMBEDDING_MODEL_CREATION] 创建嵌入模型 | model_name: ' + model_name)
    
    model = DashScopeEmbeddings(
        model=config['embedding_model_name'],
        dashscope_api_key=config['qwen_api_key']
    )
    
    app_logger.info('[EMBEDDING_MODEL_CREATED] 嵌入模型创建成功 | model_name: ' + model_name)
    return model