from chatopenai_reasoner import ChatOpenAIReasoner


def main():
    llm = ChatOpenAIReasoner(
        model="google/gemma-4-e4b",
        openai_api_key="lm-studio",
        openai_api_base="http://localhost:1234",
        lmstudio_native=True,
        lmstudio_reasoning="on",
        stream_reasoning=True,
    )
    for chunk in llm.stream("Please write a bubble sort algorithm in Python and explain the reasoning steps."):
        rc = chunk.additional_kwargs.get("reasoning_content", "")
        if rc:
            print(rc, end="", flush=True)
        else:
            print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    main()
