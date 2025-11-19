import os
from openai import OpenAI
from pathlib import Path

# 设置 OpenAI 客户端
client = OpenAI(
    api_key="",  #  API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 提示词目录
PROMPT_DIR = Path("prompts")

# 输出文件夹
output_folder = "llmGeneration"
os.makedirs(output_folder, exist_ok=True)

# 遍历每个提示词文件并生成 alpha 公式
for prompt_file in os.listdir(PROMPT_DIR):
    if prompt_file.endswith(".txt"):
        company_name = prompt_file.split("_")[1].replace(".txt", "").capitalize()  # 获取公司名
        prompt_path = PROMPT_DIR / prompt_file

        # 读取提示词文件内容
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()

        # 调用 OpenAI API 进行生成
        print(f"Generating alpha formulas for {company_name}...")
        completion = client.chat.completions.create(
            model="qwen3-max",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_content},  # 提示词内容
            ],
            stream=True  # 启用流式响应
        )

        # 用于保存文件的路径
        output_file_path = os.path.join(output_folder, f"generated_alpha_{company_name.lower()}.txt")

        # 获取模型返回的内容并写入文件
        with open(output_file_path, "w", encoding="utf-8") as f:
            for chunk in completion:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)  # 在控制台打印结果
                f.write(content)  # 保存到文件

        print(f"Generated alpha formulas for {company_name} saved to: {output_file_path}")
