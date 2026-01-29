"""
LLM Council module - implements the three-stage board flow for collaborative question generation.
Inspired by karpathy/llm-council but adapted for question generation domain.

When the chairman is Param (param-1-2.9b-instruct / rag-piped-param-instruct), a different flow is used:
Param-as-orchestrator: members generate one question each; Param only selects the best one.
This avoids Param's limitations on long-form generation.
"""
import asyncio
from typing import List, Dict, Optional, Tuple
from model_runner import run_model, needs_rag, get_rag_context

# Param models used as chairman only orchestrate (members generate, Param selects).
PARAM_ORCHESTRATOR_IDS = ("param-1-2.9b-instruct", "rag-piped-param-instruct")

def is_param_orchestrator(model_id: str) -> bool:
    """True if this model should act as chairman-orchestrator (collect from members, not generate)."""
    return model_id in PARAM_ORCHESTRATOR_IDS


def build_member_generate_one_prompt(subject: str, chapter: str, theme: str, qType: str,
                                      depth: str, language: str,
                                      topic_chunk: str = None,
                                      theme_chunk: str = None) -> str:
    """Build the prompt for a board member to generate exactly one question (used in Param-orchestrator flow)."""
    lang = (language or "en").lower()
    if lang == "hi":
        lang_block = (
            "### OUTPUT LANGUAGE\n"
            "Write the entire Question and Answer in Hindi (Devanagari script) while preserving LaTeX/math notation as-is.\n\n"
        )
    else:
        lang_block = (
            "### OUTPUT LANGUAGE\n"
            "Write the entire Question and Answer in English.\n\n"
        )
    rag_block = ""
    if topic_chunk and theme_chunk:
        rag_block = (
            "\n\n### SOURCE MATERIAL (RAG CONTEXT)\n"
            f"{topic_chunk}\n\n"
        )
    return (
        "### ROLE\n"
        "Act as an expert Academic Assessment Designer specializing in NCERT/CBSE curriculum development. "
        "Your goal is to create one high-quality question that tests true cognitive depth.\n\n"
        "### COGNITIVE DEPTH CONTEXT (Bloom's Taxonomy x DOK)\n"
        "- DOK 1 (Recall/Remember): Recall of a fact, term, or property.\n"
        "- DOK 2 (Skills & Concepts/Understand & Apply): Use of information or conceptual knowledge.\n"
        "- DOK 3 (Strategic Thinking/Analyze & Evaluate): Reasoning, planning, and using evidence.\n"
        "- DOK 4 (Extended Thinking/Create): Complex synthesis and connection across chapters.\n\n"
        "### PARAMETERS\n"
        f"- SUBJECT: {subject}\n"
        f"- CHAPTER: {chapter}\n"
        f"- THEME: {theme}\n"
        f"- QUESTION TYPE: {qType}\n"
        f"- TARGET DEPTH: {depth}\n"
        f"- QUANTITY: 1 (generate exactly one question)\n"
        f"{rag_block}"
        f"{lang_block}"
        "### CONSTRAINTS\n"
        "1. Content must be strictly based on NCERT syllabus standards.\n"
        "2. Distractors for MCQs must be 'Common Misconceptions'.\n"
        "3. For numericals, provide a step-by-step logical breakdown in the Answer section.\n"
        "4. Use LaTeX for all mathematical formulas and chemical equations (e.g., $E=mc^2$).\n\n"
        "### OUTPUT FORMAT (FOLLOW EXACTLY)\n"
        "Output exactly one question in this structure:\n"
        "<Question>\n[Question text here. If MCQ, include options A, B, C, D]\n</Question>\n"
        "<Answer>\n[Correct answer with a 2-sentence explanation of the underlying concept]\n</Answer>"
    )


def build_chairman_proposal_prompt(subject: str, chapter: str, theme: str, qType: str, 
                                   depth: str, num_questions: int, language: str,
                                   topic_chunk: str = None, 
                                   theme_chunk: str = None) -> str:
    """Build the prompt for the chairman's initial proposal."""
    lang = (language or "en").lower()
    if lang == "hi":
        lang_block = (
            "### OUTPUT LANGUAGE (CRITICAL — FOLLOW STRICTLY)\n"
            "You MUST write all Questions and Answers ONLY in Hindi (Devanagari script). Do not use English for question or answer text. Keep LaTeX/math as-is (e.g. $E=mc^2$).\n\n"
        )
        lang_reminder = "\nRemember: Output the Question and Answer in Hindi (Devanagari) only. No English.\n\n"
    else:
        lang_block = (
            "### OUTPUT LANGUAGE\n"
            "Write all Questions and Answers in English.\n\n"
        )
        lang_reminder = ""

    rag_context = ""
    if topic_chunk and theme_chunk:
        rag_context = (
            "\n\n### SOURCE MATERIAL (RAG CONTEXT)\n"
            f"{topic_chunk}\n\n"
        )
    
    prompt = (
        f"{lang_block}"
        "### ROLE\n"
        "You are the Chairman of an LLM Board responsible for generating high-quality academic assessment questions. "
        "Your role is to propose initial question drafts that will be reviewed by board members.\n\n"
        
        "### COGNITIVE DEPTH CONTEXT (Bloom's Taxonomy x DOK)\n"
        "You must adhere to the following definitions for the requested DEPTH:\n"
        "- DOK 1 (Recall/Remember): Recall of a fact, term, or property. (e.g., Define, List, State)\n"
        "- DOK 2 (Skills & Concepts/Understand & Apply): Use of information or conceptual knowledge. (e.g., Describe, Classify, Solve routine problems)\n"
        "- DOK 3 (Strategic Thinking/Analyze & Evaluate): Reasoning, planning, and using evidence. (e.g., Explain why, Non-routine problem solving, Compare/Contrast phenomena)\n"
        "- DOK 4 (Extended Thinking/Create): Complex synthesis and connection across chapters. (e.g., Create a model, Design an experiment, Critique a theoretical framework)\n\n"
        
        "### PARAMETERS\n"
        f"- SUBJECT: {subject}\n"
        f"- CHAPTER: {chapter}\n"
        f"- THEME: {theme}\n"
        f"- QUESTION TYPE: {qType}\n"
        f"- TARGET DEPTH: {depth}\n"
        f"- QUANTITY: {num_questions}\n"
        f"{rag_context}"
        
        "### CONSTRAINTS\n"
        "1. Content must be strictly based on NCERT syllabus standards.\n"
        "2. Distractors for MCQs must be 'Common Misconceptions'—they should look correct to a student who has not understood the core concept.\n"
        "3. For numericals, provide a step-by-step logical breakdown in the Answer section.\n"
        "4. Use LaTeX for all mathematical formulas and chemical equations (e.g., $E=mc^2$).\n\n"
        
        f"{lang_reminder}"
        "### OUTPUT FORMAT (FOLLOW EXACTLY)\n"
        "Generate each question in the following structure. Repeat this block for every question:\n"
        "<Question>\n[Question text here. If MCQ, include options A, B, C, D]\n</Question>\n"
        "<Answer>\n[Correct answer with a 2-sentence explanation of the underlying concept]\n</Answer>"
    )
    return prompt


def build_member_review_prompt(subject: str, chapter: str, theme: str, qType: str, 
                                depth: str, language: str,
                                chairman_proposal: str, member_letter: str,
                                topic_chunk: str = None, theme_chunk: str = None) -> str:
    """Build the prompt for a board member to review the chairman's proposal."""
    lang = (language or "en").lower()
    if lang == "hi":
        lang_block = (
            "### OUTPUT LANGUAGE (CRITICAL — FOLLOW STRICTLY)\n"
            "You MUST write your feedback, rating, and any alternative question ONLY in Hindi (Devanagari script). Do not use English.\n\n"
        )
        lang_reminder = "\nRemember: Respond in Hindi (Devanagari) only. No English.\n\n"
    else:
        lang_block = (
            "### OUTPUT LANGUAGE\n"
            "Write your feedback and any alternative question in English.\n\n"
        )
        lang_reminder = ""

    rag_context = ""
    if topic_chunk and theme_chunk:
        rag_context = (
            "\n\n### SOURCE MATERIAL (RAG CONTEXT)\n"
            f"{topic_chunk}\n\n"
        )
    
    prompt = (
        f"{lang_block}"
        "### ROLE\n"
        f"You are Board Member {member_letter} of an LLM Board. Your role is to critically review "
        "the Chairman's proposed questions and provide constructive feedback.\n\n"
        
        "### CONTEXT\n"
        f"- SUBJECT: {subject}\n"
        f"- CHAPTER: {chapter}\n"
        f"- THEME: {theme}\n"
        f"- QUESTION TYPE: {qType}\n"
        f"- TARGET DEPTH: {depth}\n"
        f"{rag_context}"
        
        "### CHAIRMAN'S PROPOSAL\n"
        "The Chairman has proposed the following question(s):\n"
        f"{chairman_proposal}\n\n"
        
        "### YOUR TASK\n"
        "1. Evaluate the quality, accuracy, and appropriateness of the proposed question(s).\n"
        "2. Rate the proposal on a scale of 1-10 (where 10 is excellent).\n"
        "3. Provide specific feedback:\n"
        "   - What works well?\n"
        "   - What could be improved?\n"
        "   - Are there any factual errors or concerns?\n"
        "   - Does it meet the requested depth level?\n"
        "4. Optionally, suggest an alternative phrasing or improvement.\n\n"
        
        f"{lang_reminder}"
        "### OUTPUT FORMAT\n"
        "Provide your review in the following format:\n"
        "<Rating>X</Rating>\n"
        "<Feedback>\n[Your detailed feedback here]\n</Feedback>\n"
        "<Alternative>\n[Optional: Your improved version of the question, or 'None' if you agree with the proposal]\n</Alternative>"
    )
    return prompt


def build_chairman_synthesis_prompt(subject: str, chapter: str, theme: str, qType: str,
                                     depth: str, language: str,
                                     original_proposal: str, member_reviews: List[Dict],
                                     topic_chunk: str = None, theme_chunk: str = None) -> str:
    """Build the prompt for the chairman to synthesize final questions based on member feedback."""
    lang = (language or "en").lower()
    if lang == "hi":
        lang_block = (
            "### OUTPUT LANGUAGE (CRITICAL — FOLLOW STRICTLY)\n"
            "You MUST write all final Questions and Answers ONLY in Hindi (Devanagari script). Do not use English for question or answer text.\n\n"
        )
        lang_reminder = "\nRemember: Output the final Question(s) and Answer(s) in Hindi (Devanagari) only. No English.\n\n"
    else:
        lang_block = (
            "### OUTPUT LANGUAGE\n"
            "Write all final Questions and Answers in English.\n\n"
        )
        lang_reminder = ""

    rag_context = ""
    if topic_chunk and theme_chunk:
        rag_context = (
            "\n\n### SOURCE MATERIAL (RAG CONTEXT)\n"
            f"{topic_chunk}\n\n"
        )
    
    reviews_text = ""
    for i, review in enumerate(member_reviews):
        reviews_text += (
            f"\n### Board Member {chr(65 + i)} Review:\n"
            f"Rating: {review.get('rating', 'N/A')}/10\n"
            f"Feedback: {review.get('feedback', 'No feedback provided')}\n"
        )
        if review.get('alternative') and review['alternative'].lower() != 'none':
            reviews_text += f"Alternative Suggestion: {review['alternative']}\n"
    
    prompt = (
        f"{lang_block}"
        "### ROLE\n"
        "You are the Chairman of an LLM Board. You have received feedback from board members on your initial proposal. "
        "Your task is to synthesize the best possible final question(s) based on this collective input.\n\n"
        
        "### CONTEXT\n"
        f"- SUBJECT: {subject}\n"
        f"- CHAPTER: {chapter}\n"
        f"- THEME: {theme}\n"
        f"- QUESTION TYPE: {qType}\n"
        f"- TARGET DEPTH: {depth}\n"
        f"{rag_context}"
        
        "### YOUR ORIGINAL PROPOSAL\n"
        f"{original_proposal}\n\n"
        
        "### BOARD MEMBER REVIEWS\n"
        f"{reviews_text}\n\n"
        
        "### YOUR TASK\n"
        "1. Consider all feedback from board members.\n"
        "2. Synthesize the best possible question(s) that incorporates valid suggestions.\n"
        "3. Ensure the final question(s) meet all requirements (depth, accuracy, format).\n"
        "4. If multiple members suggested improvements, integrate the best elements.\n\n"
        
        f"{lang_reminder}"
        "### OUTPUT FORMAT (FOLLOW EXACTLY)\n"
        "Generate the final question(s) in the following structure. Repeat this block for every question:\n"
        "<Question>\n[Question text here. If MCQ, include options A, B, C, D]\n</Question>\n"
        "<Answer>\n[Correct answer with a 2-sentence explanation of the underlying concept]\n</Answer>"
    )
    return prompt


def build_param_select_one_prompt(member_outputs: List[str]) -> str:
    """Build a short prompt for Param (chairman) to select the best one from member-generated questions."""
    parts = []
    for i, out in enumerate(member_outputs):
        parts.append(f"### Board Member {chr(65 + i)} question:\n{out}\n")
    return (
        "### ROLE\n"
        "You are the Chairman of an LLM Board. Your board members have each submitted one question. "
        "Your task is to select the single best question and output it exactly as given (do not rewrite).\n\n"
        "### BOARD MEMBER SUBMISSIONS\n"
        + "\n".join(parts) +
        "\n### YOUR TASK\n"
        "Output the single best question in this exact format (copy one member's block):\n"
        "<Question>\n[question text]\n</Question>\n"
        "<Answer>\n[answer text]\n</Answer>\n"
        "If in doubt, choose the first member's question. Output nothing else."
    )

def build_param_verify_and_select_prompt(language: str, topic_chunk: str, theme_chunk: str, member_outputs: List[str]) -> str:
    """
    Prompt Param to verify faithfulness to retrieved context (en/hi) and select the best Q&A.
    Keeps Param's generation short: choose and copy one member block.
    """
    lang = (language or "en").lower()
    lang_hint = "English" if lang == "en" else "Hindi"

    parts = []
    for i, out in enumerate(member_outputs):
        parts.append(f"### Board Member {chr(65 + i)} submission:\n{out}\n")

    return (
        "### ROLE\n"
        "You are the Chairman of an LLM Board. Your job is to verify which submission is most faithful to the provided source material "
        f"({lang_hint}) and then select the best one.\n\n"
        "### SOURCE MATERIAL (RAG CONTEXT)\n"
        f"{topic_chunk}\n\n"
        "### OPTIONAL THEME CONTEXT\n"
        f"{theme_chunk}\n\n"
        "### BOARD MEMBER SUBMISSIONS\n"
        + "\n".join(parts) +
        "\n### YOUR TASK\n"
        "1. Pick the ONE submission that is most on-topic and factually supported by the source material.\n"
        "2. Output ONLY that chosen submission, copied verbatim, in the exact format:\n"
        "<Question>\n...\n</Question>\n<Answer>\n...\n</Answer>\n\n"
        "If none are clearly supported, choose the first submission. Output nothing else."
    )

def parse_member_review(raw_output: str) -> Dict:
    """Parse a board member's review output to extract rating, feedback, and alternative."""
    import re
    
    rating_match = re.search(r'<Rating>(.*?)</Rating>', raw_output, re.DOTALL)
    feedback_match = re.search(r'<Feedback>(.*?)</Feedback>', raw_output, re.DOTALL)
    alternative_match = re.search(r'<Alternative>(.*?)</Alternative>', raw_output, re.DOTALL)
    
    rating = None
    if rating_match:
        try:
            rating = int(rating_match.group(1).strip())
        except:
            # Try to extract number from text
            num_match = re.search(r'(\d+)', rating_match.group(1))
            if num_match:
                rating = int(num_match.group(1))
    
    feedback = feedback_match.group(1).strip() if feedback_match else raw_output
    alternative = alternative_match.group(1).strip() if alternative_match else "None"
    
    return {
        "rating": rating,
        "feedback": feedback,
        "alternative": alternative,
        "raw_output": raw_output
    }


async def run_param_orchestrator_flow(chairman_model_id: str, member_model_ids: List[str],
                                       language: str,
                                       subject: str, chapter: str, theme: str, qType: str,
                                       depth: str) -> Dict:
    """
    Param-as-orchestrator: members each generate one question; Param (chairman) only selects the best.
    Always generates 1 question. Param does not generate content—only picks from member outputs.
    """
    language = (language or "en").lower()
    # Always retrieve RAG context for Param orchestration (used for member grounding + Param verification)
    loop = asyncio.get_event_loop()
    topic_chunk, theme_chunk, topic_meta, theme_meta = await loop.run_in_executor(
        None,
        lambda: get_rag_context(chapter, theme, language=language)
    )
    max_chunk = 800
    if topic_chunk and len(topic_chunk) > max_chunk:
        topic_chunk = topic_chunk[:max_chunk] + "... [truncated]"
    if theme_chunk and len(theme_chunk) > max_chunk:
        theme_chunk = theme_chunk[:max_chunk] + "... [truncated]"

    # All members generate from the same grounded prompt (1 question only)
    member_prompt = build_member_generate_one_prompt(
        subject, chapter, theme, qType, depth, language, topic_chunk=topic_chunk, theme_chunk=theme_chunk
    )

    # Stage 1: Each member generates one question (parallel)
    member_tasks = []
    for mid in member_model_ids:
        member_tasks.append(run_model(mid, member_prompt, None))
    member_outputs = await asyncio.gather(*member_tasks)

    # Stage 2: Param verifies against source and selects the best one
    verify_select_prompt = build_param_verify_and_select_prompt(
        language=language,
        topic_chunk=topic_chunk or "",
        theme_chunk=theme_chunk or "",
        member_outputs=member_outputs
    )
    final_output = await run_model(chairman_model_id, verify_select_prompt, None)

    # Member "opinions" for API compatibility: treat each member output as their submission
    member_opinions = [
        {"model_id": mid, "rating": None, "feedback": out, "alternative": "None", "raw_output": out}
        for mid, out in zip(member_model_ids, member_outputs)
    ]

    source_meta = None
    if topic_meta and len(topic_meta) > 0 and topic_meta[0]:
        source_meta = topic_meta[0]
    elif theme_meta and len(theme_meta) > 0 and theme_meta[0]:
        source_meta = theme_meta[0]
    return {
        "chairman_proposal": "[Param orchestrator: retrieved context, members generated, chairman verified+selected one.]",
        "member_opinions": member_opinions,
        "final_output": final_output,
        "source_chunks": {"topic_chunk": topic_chunk, "theme_chunk": theme_chunk},
        "source_meta": source_meta,
    }


async def run_council_flow(chairman_model_id: str, member_model_ids: List[str],
                          language: str,
                          subject: str, chapter: str, theme: str, qType: str,
                          depth: str, num_questions: int) -> Dict:
    """
    Execute the three-stage council flow for question generation.
    When chairman is Param (param-1-2.9b-instruct / rag-piped-param-instruct), uses Param-as-orchestrator
    flow instead: members generate one question each, Param selects the best (1 question only).
    
    Returns:
        Dictionary with:
        - chairman_proposal: Original proposal text (or orchestrator message)
        - member_opinions: List of member reviews or submissions
        - final_output: Raw synthesis/output
    """
    # Param as chairman: use orchestrator flow (1 question, members generate, Param selects)
    if is_param_orchestrator(chairman_model_id) and member_model_ids:
        return await run_param_orchestrator_flow(
            chairman_model_id=chairman_model_id,
            member_model_ids=member_model_ids,
            language=language,
            subject=subject,
            chapter=chapter,
            theme=theme,
            qType=qType,
            depth=depth
        )

    # Check if any model needs RAG
    needs_rag_context = needs_rag(chairman_model_id) or any(needs_rag(mid) for mid in member_model_ids)
    
    language = (language or "en").lower()
    if language == "hi":
        print("[Council] Language=Hindi; prompts will enforce Hindi (Devanagari) output.")
    topic_chunk = None
    theme_chunk = None
    topic_meta = []
    theme_meta = []
    if needs_rag_context:
        # RAG context retrieval might be blocking, run in executor
        loop = asyncio.get_event_loop()
        topic_chunk, theme_chunk, topic_meta, theme_meta = await loop.run_in_executor(
            None,
            lambda: get_rag_context(chapter, theme, language=language)
        )
    
    context_chunks = (topic_chunk, theme_chunk) if topic_chunk and theme_chunk else None
    source_meta = None
    if needs_rag_context and topic_meta and len(topic_meta) > 0 and topic_meta[0]:
        source_meta = topic_meta[0]
    elif needs_rag_context and theme_meta and len(theme_meta) > 0 and theme_meta[0]:
        source_meta = theme_meta[0]
    
    # Stage 1: Chairman proposal
    chairman_prompt = build_chairman_proposal_prompt(
        subject, chapter, theme, qType, depth, num_questions, language, topic_chunk, theme_chunk
    )
    
    # Build RAG prompt for RAG models
    if needs_rag(chairman_model_id) and context_chunks:
        # For RAG models, we need to use the RAG-specific prompt format
        _lang = (language or "en").lower()
        _rag_lang = (
            "### OUTPUT LANGUAGE (CRITICAL)\n"
            "You MUST write all Questions and Answers in Hindi only (Devanagari script). Do not use English. Keep LaTeX as-is.\n\n"
        ) if _lang == "hi" else (
            "### OUTPUT LANGUAGE\n"
            "Write all Questions and Answers in English.\n\n"
        )
        chairman_prompt = (
            f"{_rag_lang}"
            "### ROLE\n"
            "Act as an expert NCERT Assessment Designer. Your task is to use the provided 'Source Material' "
            "to generate high-quality questions. You must strictly adhere to the requested Cognitive Depth.\n\n"
            
            "### SOURCE MATERIAL (RAG CONTEXT)\n"
            f"{topic_chunk}\n\n"
            
            "### COGNITIVE DEPTH FRAMEWORK (Bloom's x DOK)\n"
            "If the source material is simple, you must still elevate the question to meet these levels:\n"
            "- DOK 1 (Recall): Direct facts from the text. (e.g., 'What is...', 'Define...')\n"
            "- DOK 2 (Understand/Apply): Interpreting the text. (e.g., 'How does X affect Y?', 'Classify...')\n"
            "- DOK 3 (Analyze/Evaluate): Using the text to solve non-routine problems. (e.g., 'What would happen if...', 'Justify...')\n"
            "- DOK 4 (Create/Synthesis): Connecting this text to broader scientific/mathematical principles.\n\n"
            
            "### SESSION PARAMETERS\n"
            f"- SUBJECT: {subject}\n"
            f"- CHAPTER: {chapter}\n"
            f"- THEME: {theme}\n"
            f"- QUESTION TYPE: {qType}\n"
            f"- REQUIRED DEPTH: {depth}\n"
            f"- QUANTITY: {num_questions}\n\n"
            
            "### INSTRUCTIONS\n"
            "1. Use the Source Material for factual accuracy. Do not hallucinate outside NCERT bounds.\n"
            "2. THE DEPTH IS PARAMOUNT: If the depth is DOK 3, do not provide a DOK 1 recall question even if the text is short.\n"
            "3. Use LaTeX for all technical notation (e.g., $H_2O$, $\sin(\theta)$).\n\n"
            
            "### OUTPUT FORMAT (FOLLOW EXACTLY)\n"
            "Strictly wrap each question and answer pair in these tags:\n"
            "<Question> [Text + Options if MCQ] </Question>\n"
            "<Answer> [Correct Answer + 1-sentence logic based on the Source Material] </Answer>"
        )
    
    chairman_output = await run_model(chairman_model_id, chairman_prompt, context_chunks)
    
    # Stage 2: Member reviews (parallel execution)
    member_tasks = []
    for i, member_id in enumerate(member_model_ids):
        member_letter = chr(65 + i)  # A, B, C, etc.
        member_prompt = build_member_review_prompt(
            subject, chapter, theme, qType, depth, language, chairman_output, member_letter,
            topic_chunk, theme_chunk
        )
        
        # Build RAG prompt for RAG models
        if needs_rag(member_id) and context_chunks:
            topic_chunk, theme_chunk = context_chunks
            _lang = (language or "en").lower()
            _rag_lang = (
                "### OUTPUT LANGUAGE (CRITICAL)\n"
                "You MUST write your feedback and alternative in Hindi only (Devanagari script). Do not use English.\n\n"
            ) if _lang == "hi" else (
                "### OUTPUT LANGUAGE\n"
                "Write your feedback and alternative in English.\n\n"
            )
            member_prompt = (
                f"{_rag_lang}"
                "### ROLE\n"
                f"You are Board Member {member_letter} of an LLM Board. Your role is to critically review "
                "the Chairman's proposed questions and provide constructive feedback.\n\n"
                
                "### SOURCE MATERIAL (RAG CONTEXT)\n"
                f"{topic_chunk}\n\n"
                
                "### CONTEXT\n"
                f"- SUBJECT: {subject}\n"
                f"- CHAPTER: {chapter}\n"
                f"- THEME: {theme}\n"
                f"- QUESTION TYPE: {qType}\n"
                f"- TARGET DEPTH: {depth}\n\n"
                
                "### CHAIRMAN'S PROPOSAL\n"
                f"{chairman_output}\n\n"
                
                "### YOUR TASK\n"
                "1. Evaluate the quality, accuracy, and appropriateness of the proposed question(s).\n"
                "2. Rate the proposal on a scale of 1-10 (where 10 is excellent).\n"
                "3. Provide specific feedback.\n"
                "4. Optionally, suggest an alternative phrasing or improvement.\n\n"
                
                "### OUTPUT FORMAT\n"
                "<Rating>X</Rating>\n"
                "<Feedback>\n[Your detailed feedback here]\n</Feedback>\n"
                "<Alternative>\n[Optional: Your improved version, or 'None']\n</Alternative>"
            )
        
        member_tasks.append(run_model(member_id, member_prompt, context_chunks))
    
    member_outputs = await asyncio.gather(*member_tasks)
    
    # Parse member reviews
    member_opinions = []
    for i, output in enumerate(member_outputs):
        opinion = parse_member_review(output)
        opinion["model_id"] = member_model_ids[i]
        member_opinions.append(opinion)
    
    # Stage 3: Chairman synthesis
    synthesis_prompt = build_chairman_synthesis_prompt(
        subject, chapter, theme, qType, depth, language, chairman_output, member_opinions,
        topic_chunk, theme_chunk
    )
    
    # Build RAG prompt for RAG models
    if needs_rag(chairman_model_id) and context_chunks:
        topic_chunk, theme_chunk = context_chunks
        _lang = (language or "en").lower()
        _rag_lang = (
            "### OUTPUT LANGUAGE (CRITICAL)\n"
            "You MUST write all final Questions and Answers in Hindi only (Devanagari script). Do not use English.\n\n"
        ) if _lang == "hi" else (
            "### OUTPUT LANGUAGE\n"
            "Write all final Questions and Answers in English.\n\n"
        )
        synthesis_prompt = (
            f"{_rag_lang}"
            "### ROLE\n"
            "You are the Chairman of an LLM Board. Synthesize the best possible final question(s) based on board member feedback.\n\n"
            
            "### SOURCE MATERIAL (RAG CONTEXT)\n"
            f"{topic_chunk}\n\n"
            
            "### CONTEXT\n"
            f"- SUBJECT: {subject}\n"
            f"- CHAPTER: {chapter}\n"
            f"- THEME: {theme}\n"
            f"- QUESTION TYPE: {qType}\n"
            f"- TARGET DEPTH: {depth}\n\n"
            
            "### YOUR ORIGINAL PROPOSAL\n"
            f"{chairman_output}\n\n"
            
            "### BOARD MEMBER REVIEWS\n"
        )
        for i, opinion in enumerate(member_opinions):
            synthesis_prompt += (
                f"\n### Board Member {chr(65 + i)} Review:\n"
                f"Rating: {opinion.get('rating', 'N/A')}/10\n"
                f"Feedback: {opinion.get('feedback', 'No feedback provided')}\n"
            )
            if opinion.get('alternative') and opinion['alternative'].lower() != 'none':
                synthesis_prompt += f"Alternative Suggestion: {opinion['alternative']}\n"
        
        synthesis_prompt += (
            "\n### YOUR TASK\n"
            "Synthesize the best possible question(s) incorporating valid suggestions.\n\n"
            
            "### OUTPUT FORMAT (FOLLOW EXACTLY)\n"
            "<Question> [Text + Options if MCQ] </Question>\n"
            "<Answer> [Correct Answer + 1-sentence logic] </Answer>"
        )
    
    final_output = await run_model(chairman_model_id, synthesis_prompt, context_chunks)
    
    source_chunks = None
    if topic_chunk and theme_chunk:
        source_chunks = {"topic_chunk": topic_chunk, "theme_chunk": theme_chunk}
    return {
        "chairman_proposal": chairman_output,
        "member_opinions": member_opinions,
        "final_output": final_output,
        "source_chunks": source_chunks,
        "source_meta": source_meta,
    }
