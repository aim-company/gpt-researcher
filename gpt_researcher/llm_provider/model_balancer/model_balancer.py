import os
from model_loadbalancer import ModelBalancer
from colorama import Fore, Style
from langchain_core.messages import ChatMessage


def convert_messages(messages: list[dict]):
    new_messages = []
    for message in messages:
        if not isinstance(message, ChatMessage):
            message = ChatMessage(**message)
        new_messages.append(message)

    return new_messages


class AimLoadBalancerProvider:

    def __init__(self, engine_name, temperature, max_tokens="max"):
        self.llm: ModelBalancer = ModelBalancer.from_engine(
            engine_name, temperature=temperature, max_tokens=max_tokens
        )

    async def get_chat_response(self, messages, stream, websocket=None):
        messages = convert_messages(messages)

        if not stream:
            # Getting output from the model chain using ainvoke for asynchronous invoking
            output = await self.llm.ainvoke(messages)

            return output.content

        else:
            return await self.stream_response(messages, websocket)

    async def stream_response(self, messages, websocket=None):
        messages = convert_messages(messages)
        paragraph = ""
        response = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    if websocket is not None:
                        await websocket.send_json(
                            {"type": "report", "output": paragraph}
                        )
                    else:
                        print(f"{Fore.GREEN}{paragraph}{Style.RESET_ALL}")
                    paragraph = ""

        return response
