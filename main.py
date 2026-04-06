from langchain_core.prompts import ChatPromptTemplate
from chatopenai_reasoner import ChatOpenAIReasoner


def main():
    llm = ChatOpenAIReasoner(
        model="google/gemma-4-e4b",
        openai_api_key="Ryanhandsome",
        openai_api_base="http://localhost:1234/v1",
        streaming=True,
        stream_reasoning=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("user", "{input}")
    ])

    messages = prompt.format_messages(input="What is 2+2? Think step by step.")

    in_thinking = False
    for chunk in llm.stream(messages):
        rc = chunk.additional_kwargs.get("reasoning_content", "")
        if rc:
            if not in_thinking:
                print("\n=== THINKING ===", flush=True)
                in_thinking = True
            print(rc, end="", flush=True)
        if chunk.content:
            if in_thinking:
                print("\n=== ANSWER ===", flush=True)
                in_thinking = False
            print(chunk.content, end="", flush=True)



if __name__ == "__main__":
    main()
