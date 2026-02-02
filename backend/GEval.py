import json,random,ast
from typing import Dict, List
from groq import Groq
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch,requests,re
groq_id_model_mapping = {
    "groq-llama-8b": "llama-3.1-8b-instant",
    "groq-llama-70b": "llama-3.3-70b-versatile",
    "rag-piped-groq-70b": "llama-3.3-70b-versatile",
    "groq-llama-guard": "meta-llama/llama-guard-4-12b",

    # Groq – GPT OSS
    "groq-gpt-oss-120b": "openai/gpt-oss-120b",
    "groq-gpt-oss-20b": "openai/gpt-oss-20b",
}
class GEval:
    def __init__(
        self,
        model: str,
        groq_api_key: str='',
        likert_scale: List[int] = None,
    ):
        """
        Parameters
        ----------
        groq_api_key : str
            Groq API key
        model : str
            Groq model name
        likert_scale : list[int]
            Likert scale values (default: [1,2,3,4,5])
        """
        is_url = lambda s: all([urlparse(s).scheme, urlparse(s).netloc])
        self.client = Groq(api_key=groq_api_key)
        self.model_name=model
        print("MODEL NAME : ",model)
        if(is_url(model)):
            self._call_model=self._call_vllm
            self.model = model
        elif('groq' in model):
            self._call_model=self._call_groq
            self.model = groq_id_model_mapping[model]
        else:
            self._call_model=self._call_hf
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                device_map="auto"
            )
        self.likert_scale = likert_scale or [1, 2, 3, 4, 5]
        self.output_format_scale=self.generate_likert_probability_string()
        
    def generate_likert_probability_string(self):
        # Step 1: generate random positive numbers
        raw = [random.random() for _ in self.likert_scale]

        # Step 2: normalize so they sum to 1
        total = sum(raw)
        probs = [round(x / total, 2) for x in raw]

        # Step 3: adjust rounding drift to ensure exact sum = 1.00
        diff = round(1.0 - sum(probs), 2)
        probs[-1] = round(probs[-1] + diff, 2)

        # Step 4: build dict with string keys
        prob_dict = {
            str(k): v for k, v in zip(self.likert_scale, probs)
        }

        # Step 5: return pretty JSON string
        return json.dumps(prob_dict, indent=2)
    # --------------------------------------------------
    # 1. Generate Chain-of-Thought (CoT)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2. Generate Likert probabilities
    # --------------------------------------------------
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
                model_output=extracted
            else:
                model_output=model_output.replace("<OUTPUT>",'').replace('</OUTPUT>','')
                extract = lambda s: {k: float(v) for k, v in re.findall(r'"(\d+)"\s*:\s*([0-9]*\.?[0-9]+)', s)}
                model_output = extract(model_output)
        except:
            model_output = model_output.strip()

        try:
            probs = json.loads(model_output.strip())
        except:
            try:
                probs = ast.literal_eval(model_output.strip())
            except Exception as e:
                print("FAILED PROBS : ",model_output,e)
                return {1:1.0}
        return {int(k): float(v) for k, v in probs.items()}

    # --------------------------------------------------
    # 3. Compute weighted average score
    # --------------------------------------------------
    def compute_weighted_score(self, probabilities: Dict[int, float]) -> float:
        return sum(
            score * probabilities.get(score, 0.0)
            for score in self.likert_scale
        )

    # --------------------------------------------------
    # 4. Full evaluation pipeline
    # --------------------------------------------------
    def evaluate(
        self,
        task_description: str,
        evaluation_parameter: str,
        question: str,
        answer: str
    ) -> float:
        if('param' in self.model_name.lower()):
            cot =''
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

    # --------------------------------------------------
    # Internal Groq streaming call (PATCHED)
    # --------------------------------------------------
    def _call_model(self,prompt: str) -> str:
        pass 

    def _call_groq(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

        model_output = ""

        for chunk in completion:
            if (chunk.choices[0].delta.content):
                model_output += chunk.choices[0].delta.content
        return model_output.strip()

    def _call_vllm(self, prompt: str) -> str:
        data = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                }

        resp = requests.post(self.model, json=data)
        resp=resp.json()
        resp=resp['choices'][0]['message']['content']
        remove_think = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
        resp=remove_think(resp)
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

        # padding special token
        inputs = self.tokenizer.apply_chat_template(
            conversation=conversation,
            return_tensors="pt",
            add_generation_prompt=True 
        )
        inputs = inputs.to(self.model.device)

        # --- Generate output ---
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

        # Get only the generated tokens (exclude the prompt length)
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
