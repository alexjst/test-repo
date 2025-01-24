import streamlit as st
import anthropic
import json
import os

client = anthropic.Anthropic()

def load_community_standards():
    with open('Roblox-Community-Standards.md', 'r') as f:
        return f.read()

def load_examples():
    with open('examples.json', 'r') as f:
        return json.load(f)['examples']

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

def main():
    st.title("Content Moderation App")
    
    examples = load_examples()
    example_names = ["None"] + [ex["name"] for ex in examples.values()]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_example = st.selectbox("Select Example", example_names)
        
        if selected_example != "None":
            example = next((ex for ex in examples.values() if ex["name"] == selected_example))
            user_message = st.text_area("User Message (Required)", value=example["user_message"], height=100)
            system_prompt = st.text_area("System Prompt (Required)", value=example["system_prompt"], height=100)
            chat_context = st.text_area("Prior Chat Context (Optional)", value=example.get("chat_context", ""), height=100)
        else:
            user_message = st.text_area("User Message (Required)", height=100)
            system_prompt = st.text_area("System Prompt (Required)", height=100)
            chat_context = st.text_area("Prior Chat Context (Optional)", height=100)
            
        if st.button("Analyze"):
            if user_message and system_prompt:
                with st.spinner('Analyzing...'):
                    result = moderate_content(user_message, system_prompt, chat_context)
                st.text_area("Moderation Result", value=result, height=400)
            else:
                st.error("Please provide both User Message and System Prompt")
    
    with col2:
        with st.expander("Community Standards Reference", expanded=True):
            st.markdown(load_community_standards())

if __name__ == "__main__":
    main()