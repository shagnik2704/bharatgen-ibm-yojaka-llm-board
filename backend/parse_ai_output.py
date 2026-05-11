import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _extract_citation_from_answer(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract citation block from answer text and strip it from the main body.
    Pattern: "Citation: <text>" (usually at the end of the text).
    
    Returns: (answer_with_citation_removed, citation_text)
    If no citation found, returns (text, None).
    """
    if not text:
        return text, None
    
    # Match "Citation: " and capture everything after it to the end of the string
    citation_pattern = r'(?im)^Citation:\s*(.*)'
    match = re.search(citation_pattern, text, flags=re.DOTALL)
    
    if match:
        citation_text = match.group(1).strip()
        # Remove the entire citation line/block from the original answer text
        clean_text = re.sub(citation_pattern, '', text, flags=re.DOTALL).strip()
        return clean_text, citation_text if citation_text else None
    
    return text, None


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        # Make sure this next line is all on one single line in your editor!
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _extract_json_payload(raw_text: str) -> Any:
    cleaned = _strip_code_fences(raw_text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    object_match = re.search(r"\{[\s\S]*\}", cleaned)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    return None


def parse_ai_output(raw_text: str) -> List[Dict[str, Any]]:
    if not raw_text:
        return []

    parsed_json = _extract_json_payload(raw_text)
    if isinstance(parsed_json, dict):
        # Handle cases where the LLM wraps the array inside an object (e.g. {"questions": [...]})
        if isinstance(parsed_json.get("questions"), list):
            parsed_json = parsed_json["questions"]
        else:
            parsed_json = [parsed_json]

    # Branch 1: JSON Output (Used heavily by Bloom Level 2)
    if isinstance(parsed_json, list):
        normalized: List[Dict[str, Any]] = []
        for item in parsed_json:
            if not isinstance(item, dict):
                continue
                
            rubric = item.get("rubric")
            answer = item.get("answer", "")
            
            # Fallback mapping if answer is nested directly inside the rubric
            # if not answer and isinstance(rubric, dict):
            #     answer = rubric.get("answer", "")
            
            # Check for explicitly mapped citation first (from the new Bloom 2 schema)
            citation = item.get("citation")
            
            # Extract citation from the answer string if it was injected there
            answer_text, extracted_citation = _extract_citation_from_answer(str(answer or "").strip())
            
            # Prioritize the explicitly keyed citation over the regex extraction
            if not citation:
                citation = extracted_citation
            
            normalized.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "answer": answer_text.strip(),
                    "citation": citation,
                    "rubric": rubric,
                }
            )
        if normalized:
            return normalized

    # Branch 2: XML Tags Output (Used by other depth formats)
    q_pattern = r"<(?:[Qq]uestion)>(.*?)(?:</[Qq]uestion>|(?=<[Qq]uestion>|<[Aa]nswer>|$))"
    a_pattern = r"<(?:[Aa]nswer)>(.*?)(?:</[Aa]nswer>|(?=<[Qq]uestion>|<[Aa]nswer>|$))"

    questions = re.findall(q_pattern, raw_text, re.DOTALL)
    answers = re.findall(a_pattern, raw_text, re.DOTALL)

    # Absolute fallback if neither JSON nor XML parsing succeeds
    if not questions:
        return [{"question": raw_text.strip(), "answer": ""}]

    out: List[Dict[str, Any]] = []
    for i, q_raw in enumerate(questions):
        a_raw = answers[i] if i < len(answers) else ""
        a_clean = re.sub(r"</?[Aa]nswer/?>", "", a_raw).strip()
        
        # Strip citation logic safely separates the text body from the "Citation: " block
        a_text, citation = _extract_citation_from_answer(a_clean)
        
        out.append(
            {
                "question": re.sub(r"</?[Qq]uestion/?>", "", q_raw).strip(),
                "answer": a_text.strip(),
                "citation": citation,
            }
        )
        
    return out