from huggingface_hub import (InferenceClient, ChatCompletionOutputMessage)
import json
HF_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN = ""


hfclient = InferenceClient(HF_MODEL_ID, token=HF_TOKEN)



def llm_call(messages):

    try:
        response = hfclient.chat.completions.create(model=HF_MODEL_ID, messages=messages, temperature=0.0)
        result = response['choices'][0]['message']['content'].replace("```", "").replace("json", "")
        json_result = json.loads(result)
        print(result)
        return result
    except Exception as e:
        print(f"LLM call error: {e}")
        return None