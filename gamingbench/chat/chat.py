import os
import time
import random
try:
    from langchain_community.chat_models import ChatOpenAI, ChatAnyscale
except ImportError:
    ChatOpenAI = None
    ChatAnyscale = None

try:
    from langchain_community.llms import DeepInfra
except ImportError:
    DeepInfra = None

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            _gemini_client = genai.Client(api_key=api_key)
        else:
            _gemini_client = genai.Client()
    return _gemini_client


def chat_llm(messages, model, temperature, max_tokens, n, timeout, stop, return_tokens=False, chat_seed=0, thinking_budget=0):
    if "gemini" in model.lower():
        client = _get_gemini_client()
        
        system_instruction = None
        contents = []
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
            else:
                role = "user" if msg['role'] == 'user' else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg['content'])]
                    )
                )
        
        config_args = {}
        if temperature is not None:
            config_args["temperature"] = temperature
        if max_tokens is not None:
            config_args["max_output_tokens"] = max_tokens
        if system_instruction is not None:
            config_args["system_instruction"] = system_instruction
        if stop is not None:
            config_args["stop_sequences"] = [stop] if isinstance(stop, list) else [stop]
            
        if thinking_budget > 0:
            config_args["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=True,
            )
            
        config = types.GenerateContentConfig(**config_args)
        
        response_list = []
        total_completion_tokens = 0
        total_prompt_tokens = 0
        
        max_retries = 5
        initial_delay = 1.0
        backoff_factor = 2
        
        for i in range(n):
            for attempt in range(1, max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )
                    
                    response_text = ""
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if getattr(part, 'thought', False):
                                response_text += f"<GEMINI_THOUGHT>\n{part.text or ''}\n</GEMINI_THOUGHT>\n"
                            else:
                                response_text += part.text or ""
                    else:
                        response_text = response.text or ""
                        
                    response_list.append(response_text)
                    if response.usage_metadata:
                        total_prompt_tokens += response.usage_metadata.prompt_token_count or 0
                        total_completion_tokens += response.usage_metadata.candidates_token_count or 0
                    break
                except Exception as e:
                    print(f"[Gemini API] call attempt {attempt} failed: {e}")
                    if attempt == max_retries:
                        raise e
                    delay = initial_delay * (backoff_factor ** (attempt - 1)) + random.uniform(0, 0.5)
                    print(f"[Gemini API] Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    
        return {
            'generations': response_list,
            'completion_tokens': total_completion_tokens,
            'prompt_tokens': total_prompt_tokens
        }

    if model.__contains__("gpt"):
        iterated_query = False
        chat = ChatOpenAI(model_name=model,
                          openai_api_key=os.environ['OPENAI_API_KEY'],
                          temperature=temperature,
                          max_tokens=max_tokens,
                          n=n,
                          request_timeout=timeout,
                          )
    elif 'Open-Orca/Mistral-7B-OpenOrca' == model:
        iterated_query = True
        chat = ChatAnyscale(temperature=temperature,
                            anyscale_api_key=os.environ['ANYSCALE_API_KEY'],
                            max_tokens=max_tokens,
                            n=1,
                            model_name=model,
                            request_timeout=timeout)
    else:
        # deepinfra
        iterated_query = True
        chat = ChatOpenAI(model_name=model,
                          openai_api_key=os.environ['DEEPINFRA_API_KEY'],
                          temperature=temperature,
                          max_tokens=max_tokens,
                          n=1,
                          request_timeout=timeout,
                          openai_api_base="https://api.deepinfra.com/v1/openai")

    longchain_msgs = []
    for msg in messages:
        if msg['role'] == 'system':
            longchain_msgs.append(SystemMessage(content=msg['content']))
        elif msg['role'] == 'user':
            longchain_msgs.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            longchain_msgs.append(AIMessage(content=msg['content']))
        else:
            raise NotImplementedError
    if n > 1 and iterated_query:
        response_list = []
        total_completion_tokens = 0
        total_prompt_tokens = 0
        for n in range(n):
            generations = chat.generate([longchain_msgs], stop=[
                stop] if stop is not None else None)
            responses = [
                chat_gen.message.content for chat_gen in generations.generations[0]]
            response_list.append(responses[0])
            completion_tokens = generations.llm_output['token_usage']['completion_tokens']
            prompt_tokens = generations.llm_output['token_usage']['prompt_tokens']
            total_completion_tokens += completion_tokens
            total_prompt_tokens += prompt_tokens
        responses = response_list
        completion_tokens = total_completion_tokens
        prompt_tokens = total_prompt_tokens
    else:
        generations = chat.generate([longchain_msgs], stop=[
            stop] if stop is not None else None)
        responses = [
            chat_gen.message.content for chat_gen in generations.generations[0]]
        completion_tokens = generations.llm_output['token_usage']['completion_tokens']
        prompt_tokens = generations.llm_output['token_usage']['prompt_tokens']

    return {
        'generations': responses,
        'completion_tokens': completion_tokens,
        'prompt_tokens': prompt_tokens
    }
