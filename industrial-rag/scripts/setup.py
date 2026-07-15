"""
项目初始化脚本
检查环境、下载模型、创建必要目录
"""
import os
import sys
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python版本过低，需要3.11+")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """检查CUDA"""
    try:
        import torch

        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU（性能较差）")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def check_redis():
    """检查Redis"""
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        print("✓ Redis连接成功")
        return True
    except Exception as e:
        print(f"⚠️  Redis未运行: {e}")
        print("   提示: 可以稍后启动Redis，或在配置中禁用缓存")
        return False


def create_directories():
    """创建必要目录"""
    dirs = [
        "data/uploads",
        "data/processed",
        "logs",
        "models",
    ]

    for d in dirs:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {d}")


def check_models():
    """检查模型"""
    models = {
        "BGE-M3": "E:/RAG/models/bge-m3",
        "BGE-Reranker": "E:/RAG/models/bge-reranker-v2-m3",
    }

    all_exist = True
    for name, path in models.items():
        if Path(path).exists():
            print(f"✓ 模型已存在: {name}")
        else:
            print(f"❌ 模型缺失: {name} ({path})")
            all_exist = False

    if not all_exist:
        print("\n📥 下载模型:")
        print("方法1 - ModelScope (国内推荐):")
        print('  pip install modelscope')
        print(
            '  python -c "from modelscope import snapshot_download; snapshot_download(\'Xorbits/bge-m3\', cache_dir=\'E:/RAG/models/bge-m3\')"'
        )
        print(
            '  python -c "from modelscope import snapshot_download; snapshot_download(\'Xorbits/bge-reranker-v2-m3\', cache_dir=\'E:/RAG/models/bge-reranker-v2-m3\')"'
        )
        print("\n方法2 - HuggingFace (需要科学上网):")
        print("  huggingface-cli download BAAI/bge-m3 --local-dir E:/RAG/models/bge-m3")

    return all_exist


def check_api_keys():
    """检查API密钥"""
    keys = {
        "CLAUDE_API_KEY": "Claude API",
        "DEEPSEEK_API_KEY": "DeepSeek API",
        "ZHIPU_API_KEY": "智谱AI",
        "OPENAI_API_KEY": "OpenAI",
    }

    found = []
    for key, name in keys.items():
        if os.getenv(key):
            print(f"✓ {name} 已配置")
            found.append(name)

    if not found:
        print("\n⚠️  未检测到API密钥")
        print("请在项目根目录创建 .env 文件，添加至少一个API密钥：")
        print("\n# 选择一个提供商")
        print("CLAUDE_API_KEY=sk-ant-xxx")
        print("# 或")
        print("DEEPSEEK_API_KEY=sk-xxx")
        print("# 或")
        print("ZHIPU_API_KEY=xxx")
        return False

    return True


def generate_env_template():
    """生成.env模板"""
    env_path = Path(".env")
    if env_path.exists():
        print("✓ .env 文件已存在")
        return

    template = """# API密钥配置（选择一个）

# Claude API (推荐 - 质量最高)
# CLAUDE_API_KEY=sk-ant-xxx

# DeepSeek (推荐 - 性价比最高)
# DEEPSEEK_API_KEY=sk-xxx

# 智谱AI (国内稳定)
# ZHIPU_API_KEY=xxx

# OpenAI
# OPENAI_API_KEY=sk-xxx

# Redis配置（可选）
REDIS_URL=redis://localhost:6379/0

# 日志级别
LOG_LEVEL=INFO
"""

    env_path.write_text(template, encoding="utf-8")
    print("✓ 创建 .env 模板文件，请填入API密钥")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Industrial RAG System - 环境检查")
    print("=" * 60)
    print()

    # 检查Python
    if not check_python_version():
        sys.exit(1)

    # 检查CUDA
    has_cuda = check_cuda()

    # 创建目录
    print("\n📁 创建目录...")
    create_directories()

    # 检查模型
    print("\n🤖 检查模型...")
    has_models = check_models()

    # 检查Redis
    print("\n💾 检查Redis...")
    check_redis()

    # 生成.env模板
    print("\n⚙️  配置文件...")
    generate_env_template()

    # 检查API密钥
    print("\n🔑 检查API密钥...")
    has_api_keys = check_api_keys()

    # 总结
    print("\n" + "=" * 60)
    print("📊 检查结果:")
    print("=" * 60)

    issues = []
    if not has_cuda:
        issues.append("⚠️  CUDA不可用 - 性能会受影响")
    if not has_models:
        issues.append("❌ 模型未下载 - 必须下载")
    if not has_api_keys:
        issues.append("❌ API密钥未配置 - 必须配置")

    if issues:
        print("\n需要处理的问题：")
        for issue in issues:
            print(f"  {issue}")
        print("\n请参考 LAPTOP_SETUP.md 完成配置")
    else:
        print("\n✅ 环境就绪！")
        print("\n下一步:")
        print("  1. 安装依赖: pip install -e .")
        print("  2. 启动服务: uvicorn app.main:app --reload")
        print("  3. 访问文档: http://localhost:8000/docs")

    print()


if __name__ == "__main__":
    main()
