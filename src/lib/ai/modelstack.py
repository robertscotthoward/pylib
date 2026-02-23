import time
from tkinter import S
import boto3
import requests
import json
import yaml
from botocore.exceptions import ClientError
from lib.tools import *




def clean_json(sJson) -> str:
    # Remove common markdown wrappers
    if '```json' in sJson:
        S = sJson.split('```json')[1]
    if "```" in sJson:
        s = sJson.split("```")[0]
    s = s.strip()
    
    s = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    s = re.sub(r'\s+', ' ', s)  # collapse multiple spaces
    return s







class ModelStack:
    def __init__(self, config):
        self.config = config
        
    def num_tokens(self):
        return from_metric(self.config.get('context-window', '1024'))

    @staticmethod
    def from_config(model_config):
        cls = model_config.get('class')
        if cls == 'ollama':
            return OllamaModelStack(model_config)
        if cls == 'bedrock':
            return BedrockModelStack(model_config)
        raise ValueError(f"Unsupported model stack class: {cls}")
    
    def query(self, prompt, max_tokens=1024):
        raise NotImplementedError("Subclasses must implement this method.")

    def query_yes_no(self, prompt, max_tokens=1024):
        # Note: When debugging, this method may timeout in the debugger's expression evaluator
        # due to network calls to LLM APIs. Set PYDEVD_WARN_EVALUATION_TIMEOUT=10 or higher
        # in your environment to increase the debugger's evaluation timeout.
        prompt = "Only respond with 'yes' or 'no' or 'maybe' as the first word on its own line. If 'maybe', follow up with a short explanation.\n" + prompt
        answer = self.query(prompt, max_tokens=max_tokens)
        word = answer.lower().strip().splitlines()[0].split(' .')[0]
        if word in ['yes', 'no']:
            return word
        return answer
    





class OllamaModelStack(ModelStack):
    def __init__(self, config):
        super().__init__(config)
        
    def query(self, prompt, max_tokens=1024):
        OLLAMA_HOST = self.config['host']
        model = self.config['model']
        max_tokens = from_metric(self.config.get('max_tokens', max_tokens))
        url = f'{OLLAMA_HOST}/api/generate'
        payload = {
            'model': model, 
            'prompt': prompt, 
            'stream': False, 
            'max_tokens': max_tokens
        }
        if 'temperature' in self.config:
            payload['temperature'] = self.config['temperature']
        if 'top_p' in self.config:
            payload['top_p'] = self.config['top_p']
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            raise Exception(f"Request failed with status code {r.status_code}: {r.text}")
        answer = json.loads(r.text)['response']
        return answer





class BedrockModelStack(ModelStack):
    def __init__(self, config):
        super().__init__(config)
        # Initialize the Bedrock client once to avoid recreating it for every query
        region = config.get('region', 'us-east-1')
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        self._last_metadata = {}
    
    def get_last_metadata(self):
        """Get metadata from the last query (token counts, etc.)"""
        return self._last_metadata
        
    def query(self, prompt, max_tokens=None):
        """Query the model and return response with metadata."""
        model = self.config['model']
        context_window = self.config.get('context-window', 200000)
        
        # Use config value or default to 4096 to leave room for input
        requested_out = max_tokens or self.config.get('max_tokens', 4096)
        
        # Estimate input tokens (rough heuristic: ~4 characters per token)
        estimated_input_tokens = len(prompt) // 4
        
        # Ensure max_tokens doesn't exceed remaining context window
        # Leave a 500-token buffer for safety
        max_available_output = context_window - estimated_input_tokens - 500
        
        if requested_out > max_available_output:
            print(f"Warning: Requested output tokens ({requested_out}) would exceed context window.")
            print(f"Estimated input: {estimated_input_tokens} tokens, context window: {context_window}")
            print(f"Adjusting max_tokens from {requested_out} to {max_available_output}")
            requested_out = max(1024, max_available_output)  # Ensure at least 1024 tokens for output
        
        # Determine model type and build appropriate params
        is_claude = 'claude' in model.lower()
        is_deepseek = 'deepseek' in model.lower()
        is_llama = 'llama' in model.lower()
        is_nova = 'nova' in model.lower()
        is_inference_profile = model.startswith('us.')
        
        if is_claude:
            # Claude models use Anthropic format
            params = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": requested_out,
                "messages": [{"role": "user", "content": prompt}],
            }
        elif is_nova:
            # Amazon Nova models require content as array of objects with text field
            # Nova doesn't support max_tokens parameter
            params = {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
            }
        elif is_inference_profile:
            # Inference profiles use prompt format without max_tokens
            params = {
                "prompt": prompt,
            }
        elif is_llama:
            # Llama models use prompt format with max_gen_len
            params = {
                "prompt": prompt,
                "max_gen_len": requested_out,
            }
        elif is_deepseek:
            # DeepSeek models use messages format
            params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": requested_out,
            }
        else:
            # Default to generic format for other models
            params = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": requested_out,
            }
        
        # Add temperature or top_p (but not both, as some models don't support both)
        # Nova doesn't support temperature/top_p, so skip for Nova models
        if not is_nova:
            if 'temperature' in self.config:
                params['temperature'] = self.config['temperature']
            elif 'top_p' in self.config:
                params['top_p'] = self.config['top_p']

        # Use the client defined in __init__
        last_error = None
        models_to_try = [model]
        
        # If model doesn't start with "us.", also try the inference profile version
        if not model.startswith('us.'):
            models_to_try.append(f'us.{model}')
        
        for model_attempt in models_to_try:
            print(f"\n[BEDROCK] Attempting to invoke model: {model_attempt}. Length of params: {len(json.dumps(params))}")
            # print(f"{json.dumps(params)}")
            try:
                with Spy('Invoke model') as spy:
                    response = self.bedrock_client.invoke_model(
                        modelId=model_attempt,
                        body=json.dumps(params),
                        contentType='application/json',
                        accept='application/json'
                    )
                with Spy('Get response body') as spy:
                    response_body = json.loads(response['body'].read())
                print(f"[BEDROCK] Successfully invoked model: {model_attempt}")
                
                # Extract response and metadata
                response_text = None
                metadata = {}
                
                # Extract response based on model type
                if is_claude:
                    response_text = response_body['content'][0]['text']
                    metadata = response_body.get('usage', {})
                elif is_nova:
                    # Nova returns output.message.content format with content as array of objects
                    content = response_body['output']['message']['content']
                    if isinstance(content, list) and len(content) > 0:
                        # Return first element's text field
                        first_item = content[0]
                        if isinstance(first_item, dict) and 'text' in first_item:
                            response_text = first_item['text']
                        else:
                            response_text = first_item
                    else:
                        response_text = content
                    metadata = response_body.get('usage', {})
                elif is_llama:
                    # Llama returns generation
                    response_text = response_body['generation']
                    metadata = response_body.get('usage', {})
                elif is_deepseek:
                    response_text = response_body['choices'][0]['message']['content']
                    metadata = response_body.get('usage', {})
                else:
                    # Try common response formats
                    if 'content' in response_body:
                        response_text = response_body['content'][0]['text']
                    elif 'choices' in response_body:
                        # Handle both string and array content formats
                        content = response_body['choices'][0]['message']['content']
                        if isinstance(content, list):
                            response_text = content[0]['text']
                        else:
                            response_text = content
                    elif 'generation' in response_body:
                        response_text = response_body['generation']
                    elif 'generations' in response_body:
                        response_text = response_body['generations'][0]['text']
                    else:
                        response_text = str(response_body)
                    metadata = response_body.get('usage', {})
                
                # Store metadata for later retrieval
                self._last_metadata = metadata
                return response_text
                
            except ClientError as e:
                print(f"ERROR: {model_attempt}: {e}")
                last_error = e
                error_code = e.response['Error']['Code']
                error_msg = e.response['Error']['Message']
                
                if error_code == 'ResourceNotFoundException':
                    print(f"ResourceNotFoundException: Model '{model_attempt}' not found or not available in this region.")
                    print(f"Error details: {error_msg}")
                    # Don't retry for model not found errors, try next model variant
                    break
                elif error_code == 'ValidationException':
                    print(f"ValidationException: {error_msg}")
                    # If it's an on-demand error, try the inference profile version
                    if "on-demand throughput isn't supported" in error_msg:
                        print(f"On-demand not supported for {model_attempt}, will try inference profile version")
                        break
                else:
                    print(f"ClientError {error_code}: {error_msg}")
            except Exception as e:
                last_error = e
                print(f"Error invoking model: {e}")
                if isinstance(e, TimeoutError) or "timed out" in str(e).lower():
                    print("Request timed out. Consider increasing timeout or retrying.")
        
        # If all retries failed, raise the last error
        raise Exception(f"Failed to invoke model after 3 attempts. Last error: {last_error}")


class TEMPLATE_ModelStack(ModelStack):
    def __init__(self, config):
        super().__init__(config)
        
    def query(self, prompt):
        answer = "..."
        return answer




def test1():
    config = {
        'class': 'ollama',  
        'host':'http://localhost:11434',
        'model': 'tinyllama:1.1b'
    }
    modelstack = ModelStack.from_config(config)
    print(modelstack.query("What city was Benjamin Franklin born in?"))


def test2():
    config = {
        'class': 'bedrock',  
        'model': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
        "temperature": 0.7,
        "region": "us-west-1"
    }
    modelstack = ModelStack.from_config(config)
    print(modelstack.query("What city was Benjamin Franklin born in?"))


if __name__ == "__main__":
    test1()
    test2()
