import pytest

from gpt_researcher.utils.llm import create_chat_completion


@pytest.mark.asyncio
async def test_using_func():
    response = await create_chat_completion(
        model="gpt-35-turbo-16k",
        messages=[
            {"role": "system", "content": f"You are a nice chat system."},
            {"role": "user", "content": "What is your day like?"},
        ],
        temperature=0,
        llm_provider="aim-loadbalancer",
        max_tokens=None,
        stream=True,
    )

    assert isinstance(response, str)
    assert len(response) > 10
    print(response)
