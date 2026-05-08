import json, random, ast, os
from typing import Dict, List
from urllib.parse import urlparse
import torch, requests, re
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

try:
    from openai import OpenAI
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OpenAI = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

groq_id_model_mapping = {
    "groq-llama-8b": "llama-3.1-8b-instant",
    "groq-llama-70b": "llama-3.3-70b-versatile",
    "rag-piped-groq-70b": "llama-3.3-70b-versatile",
    "groq-llama-guard": "meta-llama/llama-guard-4-12b",
    'groq-qwen-32b': 'qwen/qwen3-32b',
    "groq-gpt-oss-120b": "openai/gpt-oss-120b",
    "groq-gpt-oss-20b": "openai/gpt-oss-20b",
}

ollama_id_model_mapping = {
    "ollama-gemma4-e4b": "gemma4:e4b",
    "ollama-olmo-3-7b": "olmo-3:7b",
    "ollama-phi4-mini": "phi4-mini:3.8b",
    "ollama-qwen-2b": "qwen3.5:2b",
    "ollama-gemma4-e2b": "gemma4:e2b",
    "ollama-qwen-4b": "qwen3.5:4b",
    "ollama-gemma4-31b": "gemma4:31b",
}

class GEval:
    def __init__(
        self,
        model: str = 'ollama-gemma4-31b',
        groq_api_key: str = '',
        likert_scale: List[int] = None,
    ):
        """
        Parameters
        ----------
        groq_api_key : str
            Groq API key
        model : str
            Model ID or URL
        likert_scale : list[int]
            Likert scale values (default: [1,2,3,4,5])
        """
        is_url = lambda s: all([urlparse(s).scheme, urlparse(s).netloc])
        self.model_name = model
        self.provider = os.getenv("LLM_PROVIDER", "groq").lower()
        
        # 1. Ollama Provider Routing
        if self.provider == "ollama" or 'ollama' in model.lower():
            if not OLLAMA_AVAILABLE:
                raise ImportError("OpenAI package required for Ollama. Run `pip install openai`")
            self._call_model = self._call_ollama
            # Fallback to stripping 'ollama-' if the explicit mapping isn't found
            self.model = ollama_id_model_mapping.get(model, model.replace('ollama-', ''))
            self.ollama_client = OpenAI(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama" 
            )
            
        # 2. VLLM / Custom URL Routing
        elif is_url(model):
            self._call_model = self._call_vllm
            self.model = model
            
        # 3. Groq Provider Routing
        elif 'groq' in model.lower():
            if not GROQ_AVAILABLE:
                raise ImportError("Groq package required. Run `pip install groq`")
            self.client = Groq(api_key=groq_api_key)
            self._call_model = self._call_groq
            self.model = groq_id_model_mapping.get(model, model)
            
        # 4. HuggingFace Local Routing
        else:
            if not HF_AVAILABLE:
                raise ImportError("Transformers package required for HF models.")
            self._call_model = self._call_hf
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                device_map="auto"
            )

        self.likert_scale = likert_scale or [1, 2, 3, 4, 5]
        self.output_format_scale = self.generate_likert_probability_string()
        
    def generate_likert_probability_string(self):
        raw = [random.random() for _ in self.likert_scale]
        total = sum(raw)
        probs = [round(x / total, 2) for x in raw]
        diff = round(1.0 - sum(probs), 2)
        probs[-1] = round(probs[-1] + diff, 2)
        prob_dict = {str(k): v for k, v in zip(self.likert_scale, probs)}
        return json.dumps(prob_dict, indent=2)

    def _default_probability_distribution(self) -> Dict[int, float]:
        if not self.likert_scale:
            return {1: 1.0}

        weight = 1.0 / len(self.likert_scale)
        probabilities = {int(score): weight for score in self.likert_scale}
        probabilities[int(self.likert_scale[-1])] += 1.0 - sum(probabilities.values())
        return probabilities

    def _coerce_probability_distribution(self, probabilities) -> Dict[int, float]:
        if isinstance(probabilities, str):
            payload = probabilities.strip()
            if not payload:
                return self._default_probability_distribution()
            try:
                probabilities = json.loads(payload)
            except Exception:
                try:
                    probabilities = ast.literal_eval(payload)
                except Exception:
                    return self._default_probability_distribution()

        if not isinstance(probabilities, dict):
            return self._default_probability_distribution()

        cleaned: Dict[int, float] = {}
        for score in self.likert_scale:
            raw_value = probabilities.get(str(score), probabilities.get(score, 0.0))
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = 0.0

            if value != value or value < 0:
                value = 0.0

            cleaned[int(score)] = value

        total = sum(cleaned.values())
        if total <= 0:
            return self._default_probability_distribution()

        return {score: value / total for score, value in cleaned.items()}

    def generate_cot(self, task_description: str, evaluation_parameter: str) -> str:
        prompt = f"""
You are an expert evaluator. 

Task:
{task_description}

Evaluation Parameter:
{evaluation_parameter}

Generate a concise chain-of-thought explaining how this task should be evaluated using the parameter.
"""
        return self._call_model(prompt)

    def generate_likert_probabilities(
        self,
        task_description: str,
        evaluation_parameter: str,
        question: str,
        answer: str,
        cot: str
    ) -> Dict[int, float]:

        scale_str = ", ".join(map(str, self.likert_scale))
        prompt = f"""
You are an expert evaluator.

Task:
{task_description}

Evaluation Parameter:
{evaluation_parameter}

Chain-of-Thought:
{cot}

Question:
{question}

Answer:
{answer}

Return ONLY valid JSON.

Return a probability distribution over the Likert scale [{scale_str}]
where probabilities sum to 1.

Format exactly like this:
<OUTPUT>
{self.output_format_scale}
</OUTPUT>
"""

        model_output = self._call_model(prompt)
        try:
            match = re.search(r"<OUTPUT>\s*(.*?)\s*</OUTPUT>", model_output, re.DOTALL)
            if match:
                extracted = match.group(1)
                model_output = extracted
            else:
                model_output = model_output.replace("<OUTPUT>", '').replace('</OUTPUT>', '')
                extract = lambda s: {k: float(v) for k, v in re.findall(r'"(\d+)"\s*:\s*([0-9]*\.?[0-9]+)', s)}
                model_output = extract(model_output)
        except:
            model_output = model_output.strip()

        return self._coerce_probability_distribution(model_output)

    def compute_weighted_score(self, probabilities: Dict[int, float]) -> float:
        return sum(score * probabilities.get(score, 0.0) for score in self.likert_scale)

    def evaluate(
        self,
        task_description: str,
        evaluation_parameter: str,
        question: str,
        answer: str
    ) -> float:
        if 'param' in self.model_name.lower():
            cot = ''
        else:
            cot = self.generate_cot(task_description, evaluation_parameter)
            
        probabilities = self.generate_likert_probabilities(
            task_description,
            evaluation_parameter,
            question,
            answer,
            cot
        )
        return self.compute_weighted_score(probabilities)

    def _call_model(self, prompt: str) -> str:
        pass 

    def _call_ollama(self, prompt: str) -> str:
        completion = self.ollama_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
            stream=False 
        )
        return completion.choices[0].message.content.strip()

    def _call_groq(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=256,
            top_p=1,
            stream=True,
            stop=None
        )

        model_output = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                model_output += chunk.choices[0].delta.content
        return model_output.strip()

    def _call_vllm(self, prompt: str) -> str:
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        }

        resp = requests.post(self.model, json=data, verify=False)
        resp = resp.json()
        resp = resp['choices'][0]['message']['content']
        remove_think = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
        resp = remove_think(resp)
        return resp.strip()
    
    def _call_hf(self, prompt: str) -> str:
        conversation = [
            {
                "content": "You are an expert who knows about NCERT syllabus of 10th standard for physics.",
                "role": "system"
            },
            {
                "content": prompt,
                "role": "user"
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            conversation=conversation,
            return_tensors="pt",
            add_generation_prompt=True 
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.6,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )

        generated_tokens = output[0][inputs.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text.strip()
        # return resp.strip()

# evaluator = GEval(
#     likert_scale=[1, 2, 3, 4, 5]  # or [1..7]
# )

# topic='physics'
# theme='cricket'
# dok_level = '1'
# question = '''
# A cricket player hits a ball with an initial speed of $20\ \text{m s}^{-1}$ at an angle of $45^\circ$ above the horizontal, as studied in Chapter 4 – Motion in a Plane. What is the horizontal component of the ball’s initial velocity?
# A) $10\ \text{m s}^{-1}$
# B) $14\ \text{m s}^{-1}$
# C) $20\ \text{m s}^{-1}$
# D) $28\ \text{m s}^{-1}$
# '''

# answer = '''The correct answer is B) $14\ \text{m s}^{-1}$. This is calculated using the formula $v_{0x}=v_0\cos\theta$, where $v_0$ is the initial velocity and $\theta$ is the angle of projection, resulting in $v_{0x}=20\ \text{m s}^{-1}\times\cos45^\circ \approx 14.1\ \text{m s}^{-1}$, which rounds to $14\ \text{m s}^{-1}$, demonstrating the application of trigonometric principles to resolve vectors in projectile motion.'''
# theme_score = evaluator.evaluate(
#     task_description=f"You are to evaluate the thematic alignment of a question. The provided theme is {theme}.",
#     evaluation_parameter="You to rate how well it is aligned on a scale of 1 to 5. A score of 1 indicates low alignemtn while a score of 5 indicates high alignment.",
#     question=question,
#     answer=answer
# )

# topic_score = evaluator.evaluate(
#     task_description=f"You are to evaluate the topic alignment of a question. The provided theme is {topic}.",
#     evaluation_parameter="You to rate how well it is aligned on a scale of 1 to 5. A score of 1 indicates low alignemtn while a score of 5 indicates high alignment.",
#     question=question,
#     answer=answer
# )

# dok_score = evaluator.evaluate(
#     task_description=f'''You are to evaluate the Depth of Knowledge alignment of a question. 
# DOK 1: Recall & Reproduction
# DOK 2: Skills & Concepts
# DOK 3: Strategic Thinking
# DOK 4: Extended Thinking

# The provided theme is {dok_level}.''',
#     evaluation_parameter="You to rate how well it is aligned on a scale of 1 to 5. A score of 1 indicates low alignemtn while a score of 5 indicates high alignment.",
#     question=question,
#     answer=answer
# )
