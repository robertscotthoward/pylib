import glob
import os
import threading
import time
import boto3
import requests
import json
import yaml
from botocore.exceptions import ClientError
from lib.tools import *




def clean_fence(s, fence = 'json') -> str:
    # Remove common markdown wrappers
    s = str(s)
    if f'```{fence}' in s:
        s = s.split(f'```{fence}')[1]
    if "```" in s:
        s = s.split("```")[0]
    s = s.strip()
    return s




def clean_json(sJson) -> str:
    # Remove common markdown fence wrappers
    from json_repair import repair_json
    s = clean_fence(sJson, 'json')
    s = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    s = re.sub(r'\s+', ' ', s)  # collapse multiple spaces
    if s[0] != '{':
        if '{' in s and '}' in s:
            # Get the part between the first { and the last }
            s = s.split('{')[1]
            s = s.rsplit('}')[0]
            s = '{' + s + '}'
        else:
            return ''
    try:
        s = repair_json(s)
    except Exception as e:
        print(f"[ERROR] Failed to repair JSON: {e}")
        return ''
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
        if cls == 'openai_compatible':
            return OpenAICompatibleModelStack(model_config)
        if cls == 'local_whisper':
            return LocalWhisperModelStack(model_config)
        raise ValueError(f"Unsupported model stack class: {cls}")
    
    def query(self, prompt, max_tokens=1024):
        raise NotImplementedError("Subclasses must implement this method.")

    def query_image(self, prompt, image_bytes, mime_type='image/jpeg', max_tokens=1024):
        """Describe or read an image. Only vision-capable backends implement this."""
        raise NotImplementedError(f"{type(self).__name__} does not support image input")

    def transcribe_file(self, path, language=None):
        """Transcribe an audio file by path. Only speech backends implement this.

        Takes a path rather than bytes because recordings run to gigabytes, and a local
        backend can stream one off disk instead of holding it all in memory.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support audio input")

    def has_upload_limit(self) -> bool:
        """True when the backend ships the file somewhere and so caps its size."""
        return True

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
                # Claude models use streaming to avoid read-timeout on large responses
                if is_claude:
                    with Spy('Invoke model (stream)') as spy:
                        stream_response = self.bedrock_client.invoke_model_with_response_stream(
                            modelId=model_attempt,
                            body=json.dumps(params),
                            contentType='application/json',
                            accept='application/json'
                        )
                    chunks = []
                    with Spy('Read stream') as spy:
                        for event in stream_response['body']:
                            chunk = json.loads(event['chunk']['bytes'])
                            if chunk.get('type') == 'content_block_delta':
                                delta = chunk.get('delta', {})
                                if delta.get('type') == 'text_delta':
                                    chunks.append(delta.get('text', ''))
                    print(f"[BEDROCK] Successfully invoked model (stream): {model_attempt}")
                    self._last_metadata = {}
                    return ''.join(chunks)

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


class OpenAICompatibleModelStack(ModelStack):
    """Works with any provider exposing an OpenAI-style /chat/completions endpoint
    (DeepInfra, OpenRouter, Together, Groq, Fireworks, a local vLLM server, etc.).
    Swap providers by changing 'base_url' / 'model' / 'api_key' in config.yaml —
    no code changes needed.
    """

    # 429 (rate limited / "engine_overloaded") and 5xx are the provider's shared
    # capacity, not a request we sent wrong, so they're worth retrying after a wait.
    # Every other status (400, 401, 404, ...) means the request itself is bad and
    # will fail again identically, so those raise immediately instead of stalling
    # a whole `convert` run on retries that can't succeed.
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_RETRY_DELAY = 5  # seconds; doubles each attempt, so 5,10,20,40,80

    def __init__(self, config):
        super().__init__(config)

    def _request(self, content, max_tokens):
        """POST one user message and return the assistant's text.

        `content` is either a plain string or the OpenAI multi-part list used for
        vision requests, so text and image calls share the same auth and error handling.
        """
        base_url = self.config['base_url'].rstrip('/')
        model = self.config['model']
        api_key = self.config.get('api_key')
        if not api_key and self.config.get('api_key_env'):
            api_key = os.environ.get(self.config['api_key_env'])
        max_tokens = from_metric(self.config.get('max_tokens', max_tokens))

        url = f'{base_url}/chat/completions'
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
        }
        if 'temperature' in self.config:
            payload['temperature'] = self.config['temperature']
        if 'top_p' in self.config:
            payload['top_p'] = self.config['top_p']

        max_retries = self.config.get('max_retries', self.DEFAULT_MAX_RETRIES)
        retry_delay = self.config.get('retry_delay', self.DEFAULT_RETRY_DELAY)

        for attempt in range(max_retries + 1):
            r = requests.post(url, headers=headers, json=payload, timeout=self.config.get('timeout', 120))
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            if r.status_code not in self.RETRYABLE_STATUS_CODES or attempt == max_retries:
                raise Exception(f"Request failed with status code {r.status_code}: {r.text}")
            wait = retry_delay * (2 ** attempt)
            print(f"[WARN] {model}: status {r.status_code}, retrying in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries}): {r.text}")
            time.sleep(wait)

    def query(self, prompt, max_tokens=1024):
        return self._request(prompt, max_tokens)

    def transcribe_file(self, path, language=None):
        """Transcribe audio via an OpenAI-compatible /audio/transcriptions endpoint.

        A different endpoint and encoding from chat: the file goes as multipart form data
        rather than JSON, so this cannot reuse _request.
        """
        with open(path, 'rb') as handle:
            audio_bytes = handle.read()
        filename = os.path.basename(path)
        prompt = self.config.get('prompt')
        base_url = self.config['base_url'].rstrip('/')
        api_key = self.config.get('api_key')
        if not api_key and self.config.get('api_key_env'):
            api_key = os.environ.get(self.config['api_key_env'])

        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        data = {'model': self.config['model']}
        if language:
            data['language'] = language
        if prompt:
            data['prompt'] = prompt

        r = requests.post(
            f'{base_url}/audio/transcriptions',
            headers=headers,
            files={'file': (filename, audio_bytes)},
            data=data,
            timeout=self.config.get('timeout', 600),
        )
        if r.status_code != 200:
            raise Exception(f"Transcription failed with status code {r.status_code}: {r.text}")
        payload = r.json()
        return payload.get('text', '') if isinstance(payload, dict) else str(payload)

    def query_image(self, prompt, image_bytes, mime_type='image/jpeg', max_tokens=1024):
        """Send an image plus a text prompt to a vision model.

        The image travels inline as a base64 data URI, which every OpenAI-compatible
        vision endpoint accepts and which avoids needing anywhere to host the file.
        """
        import base64

        encoded = base64.b64encode(image_bytes).decode('ascii')
        content = [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{encoded}'}},
        ]
        return self._request(content, max_tokens)


class LocalWhisperModelStack(ModelStack):
    """Speech-to-text with faster-whisper on the local machine.

    Chosen over a hosted endpoint for long recordings: there is no upload, no per-file
    size cap, and no per-minute charge. faster-whisper segments a long file internally,
    so a multi-hour recording needs no chunking on our side.

    Config keys: model (e.g. 'large-v3'), device ('cuda' | 'cpu' | 'auto'),
    compute_type ('float16' | 'int8_float16' | 'int8'), beam_size, vad_filter.
    """

    # One model per (name, device, compute type), shared by every caller in the process.
    # Loading large-v3 costs seconds and gigabytes of VRAM; doing it per file would
    # dominate the run.
    _models = {}
    _load_lock = threading.Lock()
    # The GPU is one resource. Threads calling in parallel would contend for VRAM and
    # finish no sooner, so transcription is serialised.
    _gpu_lock = threading.Lock()

    @staticmethod
    def _register_cuda_dlls():
        """Put the pip-installed CUDA runtime on the DLL search path.

        CTranslate2 links cuBLAS and cuDNN by name. The nvidia-*-cu12 wheels drop those
        DLLs inside site-packages, which Windows does not search, so without this the
        GPU path dies with 'cublas64_12.dll is not found' even though CUDA is present.
        """
        if os.name != 'nt':
            return
        try:
            import nvidia
        except ImportError:
            return
        for root in nvidia.__path__:
            for entry in glob.glob(os.path.join(root, '*', 'bin')):
                try:
                    os.add_dll_directory(entry)
                except OSError:
                    pass
                # add_dll_directory only covers LoadLibraryEx with the search-path flags.
                # CTranslate2 asks for 'cublas64_12.dll' by bare name, which follows the
                # classic search order, so the directory has to be on PATH as well.
                if entry not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = entry + os.pathsep + os.environ.get('PATH', '')

    def _resolve_device(self):
        device = self.config.get('device', 'auto')
        if device != 'auto':
            return device
        try:
            import ctranslate2
            return 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
        except Exception:
            return 'cpu'

    def _get_model(self):
        # Before importing faster_whisper: it pulls in ctranslate2, which resolves the
        # CUDA libraries against whatever the search path looks like at that moment.
        self._register_cuda_dlls()
        from faster_whisper import WhisperModel

        name = self.config.get('model', 'large-v3')
        device = self._resolve_device()
        compute_type = self.config.get('compute_type') or ('float16' if device == 'cuda' else 'int8')
        key = (name, device, compute_type)

        with self._load_lock:
            if key in self._models:
                return self._models[key]
            try:
                print(f"  Loading whisper '{name}' on {device} ({compute_type})...")
                self._models[key] = WhisperModel(name, device=device, compute_type=compute_type)
            except Exception as e:
                if device != 'cuda':
                    raise
                # A broken CUDA install should slow the run down, not end it.
                print(f"[WARN] GPU unavailable ({e}); falling back to CPU.")
                key = (name, 'cpu', 'int8')
                if key not in self._models:
                    self._models[key] = WhisperModel(name, device='cpu', compute_type='int8')
            return self._models[key]

    def transcribe_file(self, path, language=None):
        model = self._get_model()
        batch_size = self.config.get('batch_size', 0)
        options = dict(
            language=language or self.config.get('language'),
            beam_size=self.config.get('beam_size', 5),
            # Skips silence, which is most of a room recording and pure cost.
            vad_filter=self.config.get('vad_filter', True),
        )

        with self._gpu_lock:
            if batch_size and batch_size > 1:
                # Batching feeds many VAD-split windows through the GPU at once, which is
                # where a large card earns its keep on multi-hour recordings.
                from faster_whisper import BatchedInferencePipeline
                pipeline = BatchedInferencePipeline(model=model)
                options.pop('vad_filter', None)  # batching always segments on VAD
                segments, _info = pipeline.transcribe(str(path), batch_size=batch_size, **options)
            else:
                segments, _info = model.transcribe(str(path), **options)
            # transcribe() returns a generator; the work happens as it is consumed.
            return ''.join(segment.text for segment in segments).strip()

    def has_upload_limit(self) -> bool:
        return False


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
