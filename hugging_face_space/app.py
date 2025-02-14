import gradio as gr
from huggingface_hub import InferenceClient
import random
import time

client = InferenceClient("HuggingFaceH4/tiny-random-LlamaForCausalLM")
def respond(
    message,
    history: list[tuple[str, str]],
):
    file = open('respond_params.txt', 'r')
    parametersLineOne = file.readline().split("=")
    parametersLineTwo = file.readline().split("=")
    parametersLineThree = file.readline().split("=")
    parametersLineFour = file.readline().split("=")
    system_message=parametersLineOne[1][:-1]
    max_tokens=int(parametersLineTwo[1][:-1])
    temperature=float(parametersLineThree[1][:-1])
    top_p=float(parametersLineFour[1][:-1])
    file.close()

    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

css_string = """
.gradio-app {height: 100%; width: 100%;}
"""

demo = gr.ChatInterface(
    respond,
    css=css_string,
    )

if __name__ == "__main__":
    demo.launch()

