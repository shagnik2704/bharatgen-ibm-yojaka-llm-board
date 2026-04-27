from typing import Any

try:
    from .guardrails import GUARDRAILS_PROMPT
except ImportError:
    from guardrails import GUARDRAILS_PROMPT


def build_prompt_from_request(req: Any, chunk_text: str) -> str:
    lang = (getattr(req, "language", "en") or "en").lower()
    if lang == "hi":
        lang_rule = "Write all output in Hindi (Devanagari), keeping math/LaTeX unchanged."
    else:
        lang_rule = "Write all output in English."

    source_block = ""
    if chunk_text:
        source_block = (
            "### SOURCE MATERIAL\n"
            f"{chunk_text}\n\n"
        )

    prompt = (
        "### ROLE\n"
        "Act as an expert Academic Assessment Designer specializing in NCERT/CBSE curriculum development. "
        "Your goal is to create questions that move beyond simple memory and test true cognitive depth.\n\n"

        "\n\n### SOURCE MATERIAL (RAG CONTEXT)\n"
        f"{source_block}"

        "### GUARDRAILS\n"
        "Follow these guardrails while answering:\n"
        f"{GUARDRAILS_PROMPT.strip()}\n\n"

        "### COGNITIVE DEPTH CONTEXT (Bloom's Taxonomy x DOK)\n"
        "You must adhere to the following definitions for the requested DEPTH:\n"
        "- DOK 1 (Recall/Remember): Recall of a fact, term, or property. (e.g., Define, List, State)\n"
        "- DOK 2 (Skills & Concepts/Understand & Apply): Use of information or conceptual knowledge. (e.g., Describe, Classify, Solve routine problems)\n"
        "- DOK 3 (Strategic Thinking/Analyze & Evaluate): Reasoning, planning, and using evidence. (e.g., Explain why, Non-routine problem solving, Compare/Contrast phenomena)\n"
        "- DOK 4 (Extended Thinking/Create): Complex synthesis and connection across chapters. (e.g., Create a model, Design an experiment, Critique a theoretical framework)\n\n"

        "### PARAMETERS\n"
        f"- SUBJECT: {getattr(req, 'subject', '')}\n"
        f"- CHAPTER: {getattr(req, 'chapter', '')}\n"
        f"- QUESTION TYPE: {getattr(req, 'qType', '')}\n"
        f"- TARGET DEPTH: {getattr(req, 'depth', '')}\n"
        f"- QUANTITY: {req.num_questions}\n\n"

        f"- LANGUAGE RULE: {lang_rule}\n"
        
        "### INSTRUCTIONS\n"
        "1. Use the Source Material for factual accuracy. Do not hallucinate outside NCERT bounds.\n"
        "2. THE DEPTH IS PARAMOUNT: If the depth is DOK 3, do not provide a DOK 1 recall question even if the text is short.\n"
        "3. Use LaTeX for all technical notation (e.g., $H_2O$, $\sin(\theta)$).\n\n"

        "### CONSTRAINTS\n"
        "1. Content must be strictly based on NCERT syllabus standards.\n"
        "2. If SOURCE MATERIAL is present, ground the output in it and do not contradict it.\n"
        "3. If SOURCE MATERIAL is absent, generate from curriculum-aligned prior knowledge without citing fake sources.\n"
        "4. Distractors for MCQs must be 'Common Misconceptions'—they should look correct to a student who has not understood the core concept.\n"
        "5. For numericals, provide a step-by-step logical breakdown in the Answer section.\n"
        "6. Use LaTeX for all mathematical formulas and chemical equations (e.g., $E=mc^2$).\n\n"

        "### OUTPUT FORMAT (FOLLOW EXACTLY)\n"
        "Generate each question in the following structure. Strictly wrap each question and answer pair in these tags (repeat this block for every question:\n"
        "<Question>\n[Question text here. If MCQ, include options A, B, C, D]\n</Question>\n"
        "<Answer>\n[Correct answer with a 2-sentence explanation of the underlying concept]\n</Answer>"
    )

    return prompt
