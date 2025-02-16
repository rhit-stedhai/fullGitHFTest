import gradio as gr
from huggingface_hub import InferenceClient
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# params on load for future
model = ""
respond_params_file = ""
css_string = """
#chatbot {
    height: 100vh
    max-height: 100vh
}
"""

chat_history = [{"start up": "test reponse"}]
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

    chat_history.append({"user": message})

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
    chat_history.append({"chatbot": response})

# need to add css to make look better
with gr.Blocks(css=css_string) as demo:
    chatbot = gr.ChatInterface(respond)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/chat/")
async def chat_get():
    return {"chat history": chat_history}

gradioApp = gr.mount_gradio_app(app, demo, path="/")
if __name__ == "__main__":
    uvicorn.run(gradioApp, host="0.0.0.0", port=7860)
