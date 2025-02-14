import gradio as gr
from huggingface_hub import InferenceClient
import random
import time

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
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

# use js and css parameter when changing the look of the window on load
#with gr.Blocks(css) as demo:
demo = gr.ChatInterface(
    respond,
    css=css_string,
    )

if __name__ == "__main__":
    demo.launch()




    # additional_inputs=[
    #     gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
    #     gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
    #     gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
    #     gr.Slider(
    #         minimum=0.1,
    #         maximum=1.0,
    #         value=0.95,
    #         step=0.05,
    #         label="Top-p (nucleus sampling)",
    #     ),
    # ],