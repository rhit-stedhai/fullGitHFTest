import gradio as gr
from huggingface_hub import InferenceClient
import random
import time
import json

from fastapi import FastAPI, UploadFile, Form
import uvicorn
import asyncio

# params on load for future
model = ""
respond_params_file = ""

app = FastAPI()

chat_history = [("start up", "test reponse")]
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
def respond(
    message,
    history: list[tuple[str, str]],
    inputFile = None,
):
    # maybe put this in an on_load on something, could be why token usage makes model degrade
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

    chat_history.append(("user", message))

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
    chat_history.append(("chatbot", response))


def process_file(file):
    if file is None:
        return "No file to process"
    
    with open(file.name, "r", encoding="utf-8"):
        content = file.read()

    # if inputFile:
    #     with open(file.name, "r", encoding="utf-8"):
    #         # error, trying to read closed file
    #         content = file.read()
    #     response += f"\nFile received:\n{content}\n"
    # else:
    #     response += f"\nNo file recieved\n"
    
    return f"File received:\n{content}"


def save_chat():
    filename = "chat_history.json"
    with open(filename, "w") as f:
        json.dump(chat_history, f, indent=4)
    return filename


@app.get("/api/chat/")
async def chat_get():
    """Handles GET requests to return a chatbot response"""
    # response_text = f"Bot: You said '{message}'"
    # chat_history.append(("User", message))
    # chat_history.append(("Bot", response_text))
    # return {"response": response_text}

    # filename = "chat_history.json"
    # with open(filename, "w") as f:
    #     json.dump(chat_history, f, indent=4)
    return {"response": "here is my response"}


css_string = """
.gradio-app {height: 100%; width: 100%;}
"""
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(respond, 
                               css=css_string,
                               )
    save_button = gr.Button("Save Chat")
    file_output = gr.File()
    save_button.click(save_chat, outputs=file_output)

# Function to run both Gradio & FastAPI
async def main():
    gradio_task = asyncio.create_task(demo.launch(share=False))
    fastapi_task = asyncio.create_task(uvicorn.run(app, host="0.0.0.0", port=7861))
    await asyncio.gather(gradio_task, fastapi_task)

if __name__ == "__main__":
    asyncio.run(main())

