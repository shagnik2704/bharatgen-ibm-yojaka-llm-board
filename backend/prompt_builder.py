from typing import Any


def is_bloom_level_2(depth: str) -> bool:
    value = (depth or "").strip().lower()
    return "bloom level 2" in value or "understand" in value or "understanding" in value


def get_generation_question_count(depth: str, requested_count: int) -> int:
    return max(1, int(requested_count or 1))

def build_prompt_from_request(req: Any, chunk_text: str) -> str:
    lang = (getattr(req, "language", "en") or "en").lower()
    enable_task_keywords = bool(getattr(req, "enable_task_keywords", True))
    depth = getattr(req, "depth", "") or ""
    question_count = get_generation_question_count(depth, getattr(req, "num_questions", 1))

    rubric_total_marks=getattr(req,"rubric_marks",5)
    if rubric_total_marks == 20:
        answer_length_instruction = "The answer MUST be a comprehensive, long-form essay. It must contain a clear Introduction, multiple distinct Body Paragraphs exploring deep critical analysis, and a definitive Conclusion."
    elif rubric_total_marks == 10:
        answer_length_instruction = "The answer MUST be structured into 2 to 3 paragraphs."
    else:
        answer_length_instruction = "The answer MUST be direct, approximately 50-75 words or 4-5 sentences. It should form a single, focused paragraph that directly addresses the question without unnecessary exposition."


    if lang == "hi":
        lang_rule = "Write all output in Hindi (Devanagari), keeping math/LaTeX unchanged."
    else:
        lang_rule = "Write all output in English."

    task_keywords_instruction = (
        "4. Give tasks like 'discuss', 'differentiate', 'comment on', 'examine', 'what is the significance', 'explain', etc. where appropriate.\n"
        if enable_task_keywords
        else ""
    )

    source_block = ""
    if chunk_text:
        source_block = (
            "### SOURCE MATERIAL\n"
            f"{chunk_text}\n\n"
        )

    use_citation = bool(getattr(req, "use_citation", False))
    citation_instructions = (
        "### CITATION DISABLED\n"
        "Do not mention citations, the source material, retrieval, or phrases like 'based on source material' or 'according to the source material' in the question or answer. "
        "Write the output as a normal assessment prompt with no source attribution language.\n\n"
    )
    if use_citation:
        citation_instructions = (
            "### CITATION-BASED MODE (ENFORCE)\n"
            "You will be provided either a direct citation or a page from the learning material which includes one or more citations. A citation is such text that comes from the original work, like a dialog from the Oedipus tragedy."
            "If you get a page with several citations mixed with the educational/explanational material, recognize which parts are the proper citations."
            "You MUST select exactly one verbatim citation from the SOURCE MATERIAL. Make sure this one citation is from its beggining to end, despite possible inconsisted formatting and other processing errors. Critically evaluate the provided material to decide what is the proper citation. Dialogues can span multiple speakers and poems may have a lot of verses, etc."
            "Provide the correct answer in the `answer` field, and provide the verbatim quote in the `citation` field. "
            "Remove any parenthetical citation markers from the quoted text (e.g., remove '(p. 175)' or '(Rajan, 29)'). "
            "Do NOT include literal line breaks or newlines inside your JSON strings. Ensure the quoted text is directly supporting the answer."
            "The answer should show understanding of the citation content.\n\n"
        )

    # Determine appropriate total marks for rubric based on Bloom level
    rubric_total_marks = 10
    if is_bloom_level_2(depth):
        depth_description = "Bloom level 2 - Understanding — Explain ideas or concepts in your own words and interpret meaning."
    else:
        # Determine appropriate rubric guidance for different Bloom levels
        depth_lower = depth.lower()
        if "bloom level 1" in depth_lower or "remember" in depth_lower or "recall" in depth_lower:
            depth_description = "Bloom level 1 questions should focus on recall, recognition, and identification of facts and concepts. Rubrics should assess accuracy of factual recall."
        elif "bloom level 3" in depth_lower or "apply" in depth_lower:
            depth_description = "Bloom level 3 - Applying — Use learned knowledge or methods to solve problems in new situations.\n"
        elif "bloom level 4" in depth_lower or "analyze" in depth_lower:
            depth_description = "Bloom level 4 - Analyzing — Break information into parts to examine relationships, patterns, or structure.\n"
        elif "bloom level 5" in depth_lower or "evaluate" in depth_lower:
            depth_description = "Bloom level 5 - Evaluating — Judge, critique, or justify decisions based on criteria and evidence.\n"
        elif "bloom level 6" in depth_lower or "create" in depth_lower:
            depth_description = "Bloom level 6 - Creating — Generate new ideas, designs, solutions, or original work by combining concepts.\n"
        else:
            depth_description = "Questions should test cognitive depth appropriate to the requested level."

    prompt = (
        "### ROLE\n"
        "Act as an expert Academic Assessment Designer specializing in NCERT/CBSE curriculum development. "
        "Your goal is to create questions that comprehensively test learning at the specified cognitive level.\n\n"

        "### SOURCE MATERIAL (RAG CONTEXT)\n"
        f"{source_block}"

        "### Bloom's Taxonomy \n"
        "You must adhere to the following definitions for the requested Bloom's taxonomy level:\n"
        'Bloom level 1 - Remembering — Recall facts, definitions, formulas, or previously learned information.\n'
        'Bloom level 2 - Understanding — Explain ideas or concepts in your own words and interpret meaning.\n'
        'Bloom level 3 - Applying — Use learned knowledge or methods to solve problems in new situations.\n'
        'Bloom level 4 - Analyzing — Break information into parts to examine relationships, patterns, or structure.\n'
        'Bloom level 5 - Evaluating — Judge, critique, or justify decisions based on criteria and evidence.\n'
        'Bloom level 6 - Creating — Generate new ideas, designs, solutions, or original work by combining concepts.\n'

        "### PARAMETERS\n"
        f"- SUBJECT: {getattr(req, 'subject', '')}\n"
        f"- CHAPTER: {getattr(req, 'chapter', '')}\n"
        f"- QUESTION TYPE: {getattr(req, 'qType', '')}\n"
        f"- TARGET BLOOM'S LEVEL: {depth}\n"
        f"- QUANTITY: {question_count}\n\n"

        f"- LANGUAGE RULE: {lang_rule}\n\n"

        "### INSTRUCTIONS\n"
        "1. Use the Source Material for factual accuracy. Do not hallucinate outside IGNOU bounds.\n"
        f"2. THE DEPTH IS PARAMOUNT: {depth_description}\n"
        f"{task_keywords_instruction}"
        f"3. Generate exactly {question_count} question(s), each with a teacher-facing rubric worth {rubric_total_marks} marks in total.\n"
        "4. Keep the output strictly valid JSON only. Do not wrap it in markdown fences or add commentary.\n"
        f"5. ANSWER LENGTH: {answer_length_instruction}\n\n"
        f"{citation_instructions}"

        "### RUBRIC REQUIREMENTS\n"
        "- Each item MUST include question, answer, and rubric fields.\n"
        "- rubric MUST include answer, marks, and key_points.\n"
        f"- marks MUST be a list of marking criteria whose values sum to {rubric_total_marks}.\n"
        "- key_points MUST be a list of the essential ideas a teacher should look for when evaluating student responses.\n"
        f"- Each mark object MUST have 'criterion' (description of what's being marked) and 'marks' (points awarded, as numbers or fractions summing to {rubric_total_marks}).\n\n"

        "### OUTPUT FORMAT (STRICT JSON ONLY)\n"
        f"Return a JSON array with exactly {question_count} object(s) using this schema:\n"
        "[\n"
        "  {\n"
        '    "question": "...",\n'
        '    "answer": "...",\n'
        '    "citation": "..." (or null if no citation),\n'
        '    "rubric": {\n'
        '      "marks": [\n'
        f'        {{"criterion": "...", "marks": <marks_value>}}\n'
        "      ],\n"
        '      "key_points": ["...", "..."]\n'
        "    }\n"
        "  }\n"
        "]"
    )

    return prompt
