from langchain_ollama import ChatOllama

def get_llm(model_name:str, context_size:int, temperature:float):
    """
    Get the LLM model with the specified parameters.
    """
    return ChatOllama(model=model_name, num_ctx=context_size, temperature=temperature)