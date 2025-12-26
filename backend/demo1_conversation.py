from .openai_realtime_client import openai_chat_client


async def handle_demo1_message(user_text: str) -> str:
    """
    Простая логика для DEMO 1: текст → ответ модели.
    Потом сюда добавим Deepgram + Realtime.
    """
    system_prompt = (
        "You are a real-time voice AI assistant. "
        "Respond concisely and clearly. "
        "Assume the user is speaking to you via voice."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    reply = openai_chat_client.chat(messages)
    return reply
