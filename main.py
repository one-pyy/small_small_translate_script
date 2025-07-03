import os
import argparse
import logging
import shutil
from openai import OpenAI
from pathlib import Path
import concurrent.futures
from pitricks.utils import init_log, print_exc

lg = init_log(logging.DEBUG)

def find_doc_files(input_dir, extensions):
    """递归查找指定目录中所有符合扩展名的文件。"""
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_paths.append(Path(root) / file)
    return file_paths

def get_translation_prompt():
    """返回固定的、高质量的系统Prompt。"""
    return """
# Role: Professional Technical Translator

You are an expert translator specializing in technical documentation. Your task is to accurately translate English technical documents into Simplified Chinese. You must strictly follow all rules to ensure the integrity of the document's structure and technical accuracy.

## Core Task
Translate the content of the file provided in the user's prompt. Your output MUST perfectly mirror the input's structure, only translating the text.

## Input/Output Format
- **Input:** A single string containing one file's path and content, starting with `FILE_PATH: [path_to_file]`.
- **Output:** You must generate a single string with the exact same structure. Reproduce the `FILE_PATH:` line precisely as it appears in the input.

## Translation Rules

1.  **Target Language:** Translate all English narrative text into professional, clear, and idiomatic **Simplified Chinese**.
2.  **Preserve Formatting (CRITICAL):**
    -   DO NOT translate or alter any formatting markers.
    -   Markdown: Keep `##`, `*`, `_`, `[]()`, ` `` `, `> ` etc., unchanged.
    -   reStructuredText (RST): Keep `:param:`, `.. note::`, `.. code-block::`, etc., unchanged.
    -   Code Blocks: Leave the content inside ```python ... ``` or similar code blocks completely untouched, EXCEPT for the comments.
3.  **Preserve Technical Terms:**
    -   DO NOT translate variable names, function names, class names, module names, file paths, or command-line arguments.
    -   For common technical concepts (e.g., "repository", "commit", "pull request", "framework", "library"), use standard, widely-accepted Chinese translations (e.g., "仓库", "提交", "拉取请求", "框架", "库").
4.  **Translate Comments Only:**
    -   Inside code blocks, only translate the English comments (e.g., lines starting with `#` in Python or `//` in JavaScript).
5.  **Maintain Structure:** The final output's structure, including the `FILE_PATH:` line, must be an exact replica of the input structure.

## Example

### Input
```
FILE_PATH: docs/quickstart.md
---
# Quickstart Guide

This guide helps you set up the project.

```python
# This is a sample function
def hello_world():
    print("Hello, World!") # Print a greeting
``````

### Expected Output
```
FILE_PATH: docs/quickstart.md
---
# 快速入门指南

本指南帮助您设置项目。

```python
# 这是一个示例函数
def hello_world():
    print("Hello, World!") # 打印问候语
```
"""

def save_translated_file(response_text: str, output_path: Path):
    """解析模型的单一文件响应并保存。"""
    try:
        # 清理模型可能返回的额外包裹，如Markdown代码块
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text.split('\n', 1)[1].rsplit('\n', 1)[0]

        # 按照Prompt要求，模型会返回带文件头的完整内容，我们只取正文部分
        _header, content = cleaned_text.split('\n---\n', 1)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')
    except (ValueError, IndexError):
        lg.warning(f"无法按预期格式解析模型对 {output_path.name} 的响应，将保存完整原始响应。")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response_text, encoding='utf-8')
    except Exception as e:
        lg.error(f"保存文件 {output_path} 时出错: {e}")

def sync_directories(src_dir: Path, dst_dir: Path, extensions: list):
    """将源目录内容同步到目标目录，不覆盖已存在的文件，并跳过待翻译的文件。"""
    lg.info(f"正在同步目录结构从 '{src_dir}' 到 '{dst_dir}'...")
    for src_path in src_dir.rglob('*'):
        relative_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / relative_path
        
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
        # 只复制不同步的文件，且目标文件不存在时才复制
        elif src_path.suffix not in extensions and not dst_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

def log_translate_progress(log_level: int, total_files: int, content: str):
    """记录翻译日志"""
    log_translate_progress._index += 1
    lg.log(log_level, f"[{log_translate_progress._index}/{total_files}] {content}")

log_translate_progress._index = 0

def translate_file(file_path: Path, input_dir: Path, output_dir: Path, client: OpenAI, model: str, total_files: int):
    """翻译单个文件并处理日志记录和错误。"""
    relative_path = file_path.relative_to(input_dir)
    output_path = output_dir / relative_path

    if output_path.exists():
        log_translate_progress(logging.INFO, total_files, f"跳过已存在的文件: {output_path.name}")
        return

    try:
        content = file_path.read_text(encoding='utf-8')
        user_content = f"FILE_PATH: {relative_path.as_posix()}\n---\n{content}"

        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_translation_prompt()},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            stream=True,  # 关键参数
        )

        # 收集流式响应
        full_response_content = []
        for chunk in stream:
            lg.debug(f"接收到流式响应: {chunk}")
            # 检查 chunk.choices[0].delta.content 是否存在且不为 None
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_response_content.append(chunk.choices[0].delta.content)
        
        translated_text = "".join(full_response_content)

        save_translated_file(translated_text, output_path)

        log_translate_progress(logging.INFO, total_files, f"翻译成功: {file_path.name}")

    except Exception as e:
        log_translate_progress(logging.ERROR, total_files, f"翻译文件 {file_path.name} 时出错: {e}")
        print_exc()

def main():
    parser = argparse.ArgumentParser(description="使用LLM逐个翻译目录中的文档文件。")
    parser.add_argument("input_dir", type=str, help="包含源文档的输入目录。")
    parser.add_argument("output_dir", type=str, help="用于保存翻译后文件的输出目录。")
    parser.add_argument("--extensions", nargs='+', default=['.md', '.rst'], help="要翻译的文件扩展名列表。")
    parser.add_argument("--api-key", type=str, default="...", help="OpenAI API密钥。默认为 OPENAI_API_KEY 环境变量。")
    parser.add_argument("--base-url", type=str, default="...", help="API代理的URL。")
    parser.add_argument("--model", type=str, default="...", help="要使用的模型名称。")
    parser.add_argument("--threads", type=int, default=10, help="并发翻译的线程数。")
    
    args = parser.parse_args()

    if not args.api_key or args.api_key == "...":
        lg.error("API密钥未提供。请使用 --api-key 参数或设置 OPENAI_API_KEY 环境变量。")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        lg.error(f"输入目录 '{input_dir}' 不存在或不是一个目录。")
        return

    # 1. 同步非翻译文件和目录结构，不覆盖已存在的文件
    sync_directories(input_dir, output_dir, args.extensions)

    # 2. 查找所有需要翻译的文档文件
    files_to_translate = find_doc_files(input_dir, args.extensions)
    if not files_to_translate:
        lg.info("未找到需要翻译的文件。")
        return
    
    total_files = len(files_to_translate)
    lg.info(f"共找到 {total_files} 个待翻译文件。开始处理...")

    # 3. 初始化API客户端
    try:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    except Exception as e:
        lg.error(f"初始化OpenAI客户端失败: {e}")
        return

    # 4. 逐个翻译文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_file = {
            executor.submit(translate_file, file, input_dir, output_dir, client, args.model, total_files): file
            for file in files_to_translate
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()  # 获取结果，如果线程中发生异常，这里会重新抛出
            except Exception as exc:
                lg.error(f"处理文件 {file.name} 时生成了一个未捕获的异常: {exc}")

    lg.info("所有翻译任务已完成。")

if __name__ == "__main__":
    main()
