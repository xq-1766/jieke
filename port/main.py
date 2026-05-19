
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# API配置
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY")
ANYTHINGLLM_BASE_URL = os.getenv("ANYTHINGLLM_BASE_URL", "https://api.anythingllm.com/v1")
MODEL = os.getenv("MODEL", "qwen/qwen3-coder:free")

# 初始化客户端（延迟初始化，避免启动时出错）
client = None
if ANYTHINGLLM_API_KEY and ANYTHINGLLM_API_KEY != "your_api_key_here":
    client = OpenAI(
        api_key=ANYTHINGLLM_API_KEY,
        base_url=ANYTHINGLLM_BASE_URL
    )
else:
    print("提示：未配置API密钥，请在.env文件中配置ANYTHINGLLM_API_KEY")
    print("程序仍可启动，但实际使用时会提示错误")

# 提示工程场景模板
SCENE_TEMPLATES = {
    "通用问答": """
你是专业提示词优化师，将用户口语化提问优化为结构化专业Prompt。
优化规则：1.设定清晰角色 2.明确任务 3.输出格式规范 4.添加约束条件 5.精简指令
用户原问题：{user_input}
只输出优化后的Prompt，无多余内容。
""",
    "代码编程": """
你是资深程序员，优化代码类提问，包含：编程语言、功能需求、输入输出、代码规范、注释要求。
用户原问题：{user_input}
只输出优化后的Prompt。
""",
    "写作文案": """
你是专业文案师，优化提问，包含：风格、字数、用途、格式、语气。
用户原问题：{user_input}
只输出优化后的Prompt。
""",
    "学习教育": """
你是专业教师，优化提问，包含：知识点、讲解方式、难度、示例。
用户原问题：{user_input}
只输出优化后的Prompt。
""",
    "办公职场": """
你是职场助理，优化提问，包含：用途、格式、简洁度、正式语气。
用户原问题：{user_input}
只输出优化后的Prompt。
"""
}

def optimize_prompt(user_input, scene):
    if not user_input.strip():
        return "请输入你的问题", "请输入问题"
    
    if client is None:
        return "错误：未配置API密钥", "请在.env文件中配置有效的ANYTHINGLLM_API_KEY"

    try:
        opt_msg = SCENE_TEMPLATES[scene].format(user_input=user_input)
        opt_res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": opt_msg}],
            temperature=0.3
        )
        optimized_prompt = opt_res.choices[0].message.content.strip()

        ans_res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": optimized_prompt}],
            temperature=0.5
        )
        ai_answer = ans_res.choices[0].message.content.strip()

        return optimized_prompt, ai_answer
    except Exception as e:
        error_msg = f"发生错误：{str(e)}"
        return error_msg, "请检查API配置或网络连接"

# 网页界面
with gr.Blocks(title="AI提示词一键优化器 | 提示工程结课作品") as demo:
    gr.Markdown("# 🚀 AI提示词一键优化器（提示工程结课作品）")
    gr.Markdown("基于 AnythingLLM + Qwen3‑Coder 实现")

    scene = gr.Dropdown(label="使用场景", choices=list(SCENE_TEMPLATES.keys()), value="通用问答")
    user_input = gr.Textbox(label="原始问题（口语化）", lines=4, placeholder="例如：帮我写冒泡排序代码")
    btn = gr.Button("一键优化Prompt并生成回答", variant="primary")

    optimized_prompt = gr.Textbox(label="✅ 优化后的专业Prompt（提示工程成果）", lines=6)
    ai_answer = gr.Textbox(label="🤖 AI最终回答", lines=10)

    btn.click(optimize_prompt, inputs=[user_input, scene], outputs=[optimized_prompt, ai_answer])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

