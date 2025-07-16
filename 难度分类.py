# batch_inference_debug.py
#
# 一个高性能、健壮的大语言模型批量推理脚本
# 使用兼容OpenAI的API。
#
# 依赖库:
# pip install openai datasets httpx tqdm
#

# 导入必要的库
import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path  # 用于处理文件路径
from typing import Any, Dict, List

import httpx  # 高性能的HTTP客户端
from datasets import load_dataset  # 从Hugging Face Hub加载数据集
from openai import AsyncOpenAI  # 异步OpenAI客户端
from tqdm.asyncio import tqdm  # 异步任务的进度条

# --- 全局常量定义 ---
EXTRACT_FAIL = "EXTRACT_FAIL"  # 答案提取失败时的标记
API_ERROR = "API_ERROR"  # API调用失败时的标记

# --- 日志系统设置 ---
def setup_logging() -> logging.Logger:
    """配置并返回一个日志记录器。"""
    logger = logging.getLogger("BatchInference")
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    # 防止重复添加处理器
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)  # 将日志输出到控制台
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# 初始化日志记录器
logger = setup_logging()

# --- 核心逻辑函数 ---

def extract_answer(text: str) -> str:
    """
    从文本中提取最后一个 \boxed{} 内的内容。

    Args:
        text (str): 模型的文本输出。

    Returns:
        str: 提取到的答案，如果未找到则返回 EXTRACT_FAIL。
    """
    # 如果输入本身就是错误标记，直接返回
    if text in [API_ERROR, EXTRACT_FAIL]:
        return text
    
    # 使用正则表达式查找所有 \boxed{...} 的匹配项
    # (.*?) 是一个非贪婪匹配，用于捕获花括号内的所有内容
    matches = re.findall(r"\\boxed\{(.*?)\}", text)

    if matches:
        # 如果找到匹配项，返回最后一个匹配项的内容，并去除首尾空格
        return matches[-1].strip()
    else:
        # 如果未找到，返回提取失败的标记
        return EXTRACT_FAIL

def is_correct(model_answer: str, true_answer: str) -> bool:
    """
    为MATH数据集比较模型答案和真实答案的正确性。
    能处理数字、小数和常见的格式问题。

    Args:
        model_answer (str): 从模型输出中提取的答案。
        true_answer (str): 数据集中的标准答案。

    Returns:
        bool: 如果答案匹配则为True，否则为False。
    """
    # 如果模型答案是错误标记或None，直接判为错误
    if model_answer in [API_ERROR, EXTRACT_FAIL] or model_answer is None:
        return False

    try:
        # 规范化两个答案：转为字符串、去除首尾空格、移除逗号
        model_answer_norm = str(model_answer).strip().replace(",", "")
        true_answer_norm = str(true_answer).strip().replace(",", "")

        # 对于简单情况，直接进行字符串比较
        if model_answer_norm == true_answer_norm:
            return True
        
        # 尝试进行数值比较
        model_float = float(model_answer_norm)
        true_float = float(true_answer_norm)
        
        # 比较两个浮点数是否足够接近（处理精度问题）
        return abs(model_float - true_float) < 1e-6

    except (ValueError, TypeError):
        # 如果数值转换失败，则退回到不区分大小写的字符串比较
        return model_answer_norm.lower() == true_answer_norm.lower()

async def fetch_completion(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    payload: Dict[str, Any]
) -> str:
    """
    使用信号量控制并发，从API获取单个模型的补全结果。

    Args:
        client (AsyncOpenAI): 异步OpenAI客户端实例。
        semaphore (asyncio.Semaphore): 用于限制并发请求数量的信号量。
        payload (Dict[str, Any]): API请求的负载数据。

    Returns:
        str: 模型的响应文本，如果失败则返回 API_ERROR。
    """
    # 等待信号量，确保并发数不超过限制
    async with semaphore:
        try:
            # 发起异步API调用
            response = await client.chat.completions.create(**payload)
            # 返回消息内容，如果内容为空则返回空字符串
            return response.choices[0].message.content or ""
        except Exception as e:
            # 记录API调用失败的错误
            logger.error(f"API调用失败: {e}")
            return API_ERROR

async def process_single_item(
    item: Dict[str, Any],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: Dict[str, Any], # 使用配置字典替代argparse.Namespace
    prompt_template: str,
) -> Dict[str, Any]:
    """
    处理数据集中的单个条目，包括其所有的采样请求。

    Args:
        item (Dict[str, Any]): 代表数据集中一行数据的字典。
        client (AsyncOpenAI): 异步OpenAI客户端。
        semaphore (asyncio.Semaphore): 异步信号量。
        config (Dict[str, Any]): 包含所有脚本参数的配置字典。
        prompt_template (str): 提示词模板字符串。

    Returns:
        Dict[str, Any]: 包含该条目最终处理结果的字典。
    """
    question = item[config['question_column']]
    true_answer = item[config['true_answer_column']]
    
    #之前的版本
    # # 将问题填入提示词模板
    # prompt = prompt_template.format(question=question)
    
    # # 构建API请求的负载
    # payload = {
    #     "model": config['model_name'],
    #     "messages": [{"role": "user", "content": prompt}],
    #     "temperature": config['temperature'],
    #     "max_tokens": config['max_tokens'],
    # }

    # 【重要修改】构建包含 system 和 user 角色的 messages 列表
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": question}
    ]
    
    # 构建API请求的负载
    payload = {
        "model": config['model_name'],
        "messages": messages, # 使用新的 messages 列表
        "temperature": config['temperature'],
        "max_tokens": config['max_tokens'],
    }

    # 为所有采样创建并发任务
    tasks = [
        fetch_completion(client, semaphore, payload)
        for _ in range(config['sample_all'])
    ]
    # 并发执行所有任务并等待结果
    raw_outputs = await asyncio.gather(*tasks)

    # 处理返回的结果
    sampling_results = []
    correct_count = 0
    for i, raw_output in enumerate(raw_outputs):
        extracted = extract_answer(raw_output)
        correct = is_correct(extracted, true_answer)
        if correct:
            correct_count += 1
        
        sampling_results.append({
            "sample_id": i + 1,
            "output": raw_output,
            "extracted_answer": extracted,
            "is_correct": correct,
        })
    
    # 构建最终的JSON结果对象
    final_result = {
        key: item[key] for key in config['columns_to_keep']
    }
    final_result["sampling_results"] = sampling_results
    final_result[config['model_name']] = (correct_count >= config['sample_true'])
    
    return final_result


# --- 文件和数据处理函数 ---

def create_default_prompt_file(path: Path):
    """如果默认的提示词模板文件不存在，则创建一个。"""
    if not path.exists():
        logger.info(f"正在创建默认提示词模板: {path}")
        default_prompt = (
            "You are an expert mathematician. Please solve the following math problem.\n"
            "Think step by step and show your reasoning.\n"
            "Finally, put your final numerical answer in a single box using the format: \\boxed{answer}.\n\n"
            "Problem:\n{question}\n\n"
            "Solution:"
        )
        path.write_text(default_prompt, encoding="utf-8")

def read_and_display_results(filepath: Path):
    """
    读取输出的.jsonl文件并展示结果摘要。
    
    Args:
        filepath (Path): .jsonl结果文件的路径。
    """
    if not filepath.exists():
        logger.error(f"结果文件未找到: {filepath}")
        return

    logger.info("\n" + "="*20 + " 结果摘要 " + "="*20)
    logger.info(f"正在读取结果文件: {filepath}")

    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
            
    if not results:
        logger.warning("结果文件为空。")
        return

    total_items = len(results)
    # 动态获取模型名称（即结果中布尔类型的那个键）
    model_name_keys = [k for k in results[0].keys() if isinstance(results[0][k], bool)]
    if not model_name_keys:
        logger.error("在结果中找不到模型名称对应的成功标记键。")
        return
    model_name = model_name_keys[0]

    overall_success_count = sum(1 for r in results if r.get(model_name, False))
    overall_accuracy = (overall_success_count / total_items) * 100 if total_items > 0 else 0

    total_samples = 0
    total_correct_samples = 0
    for r in results:
        for sample in r.get('sampling_results', []):
            total_samples += 1
            if sample.get('is_correct', False):
                total_correct_samples += 1

    sample_accuracy = (total_correct_samples / total_samples) * 100 if total_samples > 0 else 0
    
    logger.info(f"评估的模型: {model_name}")
    logger.info(f"处理的总问题数: {total_items}")
    logger.info(f"整体问题成功率 (pass@k): {overall_accuracy:.2f}% ({overall_success_count}/{total_items})")
    logger.info(f"单次采样准确率 (acc@1): {sample_accuracy:.2f}% ({total_correct_samples}/{total_samples})")
    logger.info("="*57 + "\n")


# --- 主执行流程 ---

async def run_inference(config: Dict[str, Any]):
    """主函数，用于组织和运行整个推理流程。"""

    # --- 初始设置 ---
    start_time = time.time()
    logger.info("启动批量推理流程...")
    logger.info(f"配置参数: {config}")

    # 检查并创建默认提示词文件
    prompt_path = Path(config['prompt_template_path'])
    create_default_prompt_file(prompt_path)

    # 加载提示词模板
    try:
        prompt_template = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"提示词模板文件未找到: {prompt_path}")
        sys.exit(1)


    # --- 数据加载 ---
    logger.info(f"正在加载数据集 '{config['dataset_path']}' 的 '{config['dataset_split']}' 部分...")
    try:
        # 加载完整的数据集拆分
        dataset = load_dataset(config['dataset_path'], split=config['dataset_split'])
        
        # 检查是否设置了调试条目数限制
        debug_items = config.get("debug_max_items")
        if debug_items is not None and debug_items > 0:
            # 如果设置了，则只选取指定数量的条目
            dataset = dataset.select(range(min(debug_items, len(dataset))))
            logger.info(f"【调试模式】已选择前 {len(dataset)} 条数据进行处理。")
        else:
            logger.info(f"【完整运行模式】将处理全部 {len(dataset)} 条数据。")

    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        sys.exit(1)

    # --- API和输出设置 ---
    client = AsyncOpenAI(
        api_key=config['api_key'],
        base_url=config['api_base'],
        # http_client=httpx.AsyncClient(timeout=config['api_timeout'])
    )
    semaphore = asyncio.Semaphore(config['async_concurrency'])
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True) # 确保输出目录存在
    output_path = output_dir / f"{config['model_name']}.jsonl"

    # --- 运行推理 ---
    logger.info(f"开始对 {len(dataset)} 个条目进行推理...")
    tasks = [
        process_single_item(item, client, semaphore, config, prompt_template)
        for item in dataset
    ]

    results = []
    # 使用tqdm显示进度条
    for future in tqdm.as_completed(tasks, total=len(tasks), desc="处理条目中"):
        results.append(await future)

    # --- 保存结果 ---
    logger.info(f"正在将 {len(results)} 条结果保存到 {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            # 将字典转换为JSON字符串并写入文件，每条占一行
            f.write(json.dumps(result) + "\n")

    end_time = time.time()
    logger.info(f"批量推理在 {end_time - start_time:.2f} 秒内完成。")
    
    # --- 显示摘要 ---
    read_and_display_results(output_path)


if __name__ == "__main__":
    # =================================================================
    # --- 在这里修改所有参数以便调试 ---
    # =================================================================

    # 创建一个字典来存储所有配置参数
    config = {
        "model_name": "qwen3-8b",
        "api_key": 'sk-7b407980bfe347f3bf6fb8ddfeeac897',  # 替换为你的API Key
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 替换为你的API地址
        "dataset_path": "/raid_sdb/home/libingwei/datas/OpenR1-Math-220k",
        "dataset_split": "train",
        "columns_to_keep": ["problem", "answer"],
        "question_column": "problem",
        "true_answer_column": "answer",
        "prompt_template_path": "/raid_sdb/home/libingwei/CoT压缩/prompt.txt",
        "output_dir": "/raid_sdb/home/libingwei/CoT压缩/output",
        "sample_all": 3,  # 每个问题采样3次
        "sample_true": 2, # 至少答对2次才算成功
        "temperature": 0.7,
        "max_tokens": 1024,
        "api_timeout": 60,
        "async_concurrency": 8, # 并发请求数
        "debug_max_items": 2 # 【调试专用】设置处理的数据条目数，以加快调试速度
    }
    
    # 运行主异步函数
    asyncio.run(run_inference(config))