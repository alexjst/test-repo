import gradio as gr
import anthropic
import json
import os
from typing import Dict, Any

client = anthropic.Anthropic()

def load_community_standards():
    with open('Roblox-Community-Standards.md', 'r') as f:
        return f.read()

def load_example(example_name: str) -> Dict[str, str]:
    if example_name == "None":
        return {
            "user_message": "",
            "system_prompt": "",
            "chat_context": ""
        }
    with open('examples.json', 'r') as f:
        examples = json.load(f)['examples']
    
    # Find the example with matching name
    example = next((ex for ex in examples.values() if ex["name"] == example_name), None)
    if example:
        return {
            "user_message": example["user_message"],
            "system_prompt": example["system_prompt"],
            "chat_context": example.get("chat_context", "")
        }
    return {"user_message": "", "system_prompt": "", "chat_context": ""}

def moderate_content(message, system_prompt, chat_context=""):
    md_content = load_community_standards()
    
    prompt = f"""You are a content moderator. Analyze the following chat for Community Standards violations.

Context required for analysis:
- User message: "{message}"
- System prompt: "{system_prompt}"
- Prior chat context: "{chat_context}"
- Community Standards: {md_content}

Return ONLY these fields in your response, with no additional text:
Violation: [Yes/No]
Type: [violation type if any]
Policy: [specific policy section reference]
Confidence: [High/Medium/Low]
Reason: [brief explanation]
Action: [recommended action]"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

with gr.Blocks() as app:
    gr.Markdown("# Content Moderation App")
    
    with open('examples.json', 'r') as f:
        examples = json.load(f)['examples']
    
    with gr.Row():
        with gr.Column():
            example_dropdown = gr.Dropdown(
                choices=["None"] + [v["name"] for v in examples.values()],
                value="None",
                label="Select Example"
            )
            user_message = gr.Textbox(label="User Message (Required)", lines=3)
            system_prompt = gr.Textbox(label="System Prompt (Required)", lines=3)
            chat_context = gr.Textbox(label="Prior Chat Context (Optional)", lines=3)
            submit_btn = gr.Button("Analyze")
        
        with gr.Column():
            output = gr.Textbox(label="Moderation Result", lines=16)
    
    def update_inputs(example_id):
        example = load_example(example_id)
        return [
            example["user_message"],
            example["system_prompt"],
            example["chat_context"]
        ]
    
    example_dropdown.change(
        update_inputs,
        inputs=[example_dropdown],
        outputs=[user_message, system_prompt, chat_context]
    )
    
    submit_btn.click(
        moderate_content,
        inputs=[user_message, system_prompt, chat_context],
        outputs=[output]
    )
    
    gr.Markdown("## Community Standards Reference")
    standards_display = gr.Textbox(value=load_community_standards(), label="", lines=20, interactive=False)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=True)