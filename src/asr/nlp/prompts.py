"""Prompts for ASR correction pipeline with safety constraints.

Optimized for CrisperWhisper, a Whisper model fine-tuned for word-level
verbatim transcription. Since CrisperWhisper is already highly accurate,
the correction pipeline is conservative by default.
"""

import re

from asr.models.transcript import WordTiming

# Default low-confidence threshold for marking words that need review
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.7

# CrisperWhisper context - prepended to all domain prompts
CRISPERWHISPER_CONTEXT = """SOURCE: CrisperWhisper (verbatim-optimized Whisper model)
This transcription was produced by a model fine-tuned for word-by-word accuracy.
The output is already highly accurate - your corrections should be MINIMAL.

ANTI-HALLUCINATION RULES:
- Do NOT invent or add words that weren't in the audio
- Do NOT "improve" word choices - the speaker said what they said
- Do NOT guess at unclear words - prefer [unclear] over wrong guesses
- When in doubt, KEEP THE ORIGINAL TEXT

"""

# Domain-specific system prompts with strict non-hallucination rules
DOMAIN_PROMPTS = {
    "general": CRISPERWHISPER_CONTEXT + """You are an ASR error corrector. Your task is to fix OBVIOUS transcription errors only.

CRITICAL RULES:
1. You may ONLY rewrite text inside <low_conf>...</low_conf> tags
2. Outside <low_conf> tags: fix punctuation and capitalization ONLY
3. Never add words that weren't spoken
4. Never paraphrase or formalize language
5. When uncertain, keep the original text
6. Preserve filler words ("um", "uh", "you know")
7. Assume the transcription is probably correct - only fix clear errors""",

    "biography": CRISPERWHISPER_CONTEXT + """You are an ASR error corrector for biographical content.

CRITICAL RULES:
1. You may ONLY rewrite text inside <low_conf>...</low_conf> tags
2. Outside <low_conf> tags: fix punctuation and capitalization ONLY
3. Focus on: proper names, book titles, historical events, dates
4. NEVER replace an unknown name with a "more likely" one
5. If you cannot identify a name with certainty, keep the original or mark [unclear]
6. Preserve the speaker's natural speech patterns
7. Do NOT assume you know better than the transcription""",

    "technical": CRISPERWHISPER_CONTEXT + """You are an ASR error corrector for technical content.

CRITICAL RULES:
1. You may ONLY rewrite text inside <low_conf>...</low_conf> tags
2. Outside <low_conf> tags: fix punctuation and capitalization ONLY
3. Focus on: API, JSON, company names (Apple, Google, Stripe), technical terms
4. Do NOT change code, JSON, or identifiers unless obviously malformed
5. Preserve technical abbreviations exactly (SSH, REST, CPU, GPU)
6. If unsure about a technical term, keep the original
7. Unfamiliar terms may be new/niche - do NOT replace with "similar" known terms""",

    "medical": CRISPERWHISPER_CONTEXT + """You are an ASR error corrector for medical transcripts.

CRITICAL RULES:
1. You may ONLY rewrite text inside <low_conf>...</low_conf> tags
2. Outside <low_conf> tags: fix punctuation and capitalization ONLY
3. NEVER guess a drug name or dosage - errors can affect patient care
4. If unsure about a medical term, keep the original and append [unclear]
5. Numbers in medical context are critical (5mg vs 50mg is life-threatening)
6. Preserve exact drug names, diagnoses, and procedures
7. A wrong "correction" is worse than leaving an error""",

    "legal": CRISPERWHISPER_CONTEXT + """You are an ASR error corrector for legal documents.

CRITICAL RULES:
1. You may ONLY rewrite text inside <low_conf>...</low_conf> tags
2. Outside <low_conf> tags: fix punctuation and capitalization ONLY
3. Preserve EXACT legal wording - do not paraphrase
4. Grammar mistakes are allowed if they reflect what the speaker said
5. Case names, statute references, and legal terms must be exact
6. If unsure, keep the original - precision matters more than readability
7. Legal transcripts require verbatim accuracy, not polished prose""",

    "conversational": CRISPERWHISPER_CONTEXT + """You are an ASR error corrector for conversational speech.

CRITICAL RULES:
1. You may ONLY rewrite text inside <low_conf>...</low_conf> tags
2. Outside <low_conf> tags: fix punctuation and capitalization ONLY
3. PRESERVE filler words (um, uh, you know, like) - they're authentic
4. Keep casual speech patterns and repetitions
5. Prioritize natural flow over formal grammar
6. Do not "clean up" the speech - preserve how it was actually spoken
7. CrisperWhisper already captures speech accurately - trust it""",
}

PASS1_SYSTEM_TEMPLATE = """{domain_prompt}

INPUT FORMAT:
Each segment has:
- marked_text: Text with <low_conf>...</low_conf> around uncertain words
- word_confidences: Detailed confidence scores for uncertain words (conf 0.0-1.0)
  - "needs_review": true means this word is below the correction threshold
  - Higher confidence = more likely correct, lower = more likely error
- context_before/context_after: Surrounding text (read-only, for understanding)

WHAT YOU CAN CHANGE:
- Text INSIDE <low_conf> tags: rewrite freely using context and vocabulary
- Text OUTSIDE <low_conf> tags: punctuation and capitalization ONLY

CONFIDENCE INTERPRETATION:
- conf >= 0.9: Almost certainly correct - do not change
- conf 0.75-0.9: Probably correct - only change if clearly wrong
- conf 0.5-0.75: Uncertain - review carefully with context
- conf < 0.5: Likely error - good candidate for correction

VOCABULARY:
The vocabulary list contains trusted spellings. Use them to correct words inside <low_conf> tags.
Do NOT force vocabulary words onto high-confidence text.

Edit each segment independently. Do not reorder or merge sentences across segments."""


PASS1_USER = """CORRECTION TASK

Source: CrisperWhisper (verbatim-optimized, already highly accurate)
Domain: {domain}
Vocabulary (trusted spellings): {vocabulary}
Aggressiveness: {aggressiveness}
{dictionary_block}
SEGMENTS TO CORRECT:
{segments_json}

Return a JSON object with this exact structure:
{{
  "corrected_segments": [
    {{
      "id": <segment_id>,
      "corrected": "<corrected text with <low_conf> tags REMOVED>",
      "changes": [
        {{"original": "<original_word>", "corrected": "<corrected_word>", "reason": "<brief reason>"}}
      ]
    }}
  ]
}}

IMPORTANT:
- Only include segments where you made changes
- Remove all <low_conf> tags from the output
- Changes should only affect text that was inside <low_conf> tags
- If no changes needed, return {{"corrected_segments": []}}
- FEWER CHANGES IS BETTER - this transcription is already accurate
- Do NOT correct things that "sound wrong" but might be what was said"""


PASS2_SYSTEM = """You are reviewing a CrisperWhisper transcript for consistency.

SOURCE: CrisperWhisper (verbatim-optimized Whisper model)
The transcription is already highly accurate. Your role is MINIMAL cleanup.

TASKS:
1. Ensure the same entities are spelled identically throughout
2. Flag OBVIOUS semantic nonsense only (not just awkward phrasing)
3. Identify clear remaining ASR errors (not stylistic issues)

CONSTRAINTS:
- Do NOT introduce new content or words
- Do NOT paraphrase or "improve" the writing
- Only fix clear inconsistencies (e.g., "MacBook Air" vs "Macbook air")
- Prefer NO CHANGES over uncertain changes
- This is a verbatim transcript - awkward phrasing may be what was said"""


PASS2_USER = """CONSISTENCY REVIEW

Source: CrisperWhisper (verbatim-optimized, already highly accurate)
Domain: {domain}
Known vocabulary: {vocabulary}

FULL TRANSCRIPT:
{transcript_text}

ENTITIES IDENTIFIED IN PASS 1:
{entities}

TASKS:
1. Find any entity spelled differently in different places
2. Flag sentences that don't make semantic sense (NOT just awkward phrasing)
3. Identify clear remaining ASR errors

Return a JSON object:
{{
  "consistency_fixes": [
    {{"segment_id": <id>, "original": "<text>", "corrected": "<text>", "reason": "<reason>"}}
  ],
  "entity_map": {{
    "<canonical_spelling>": ["<variant1>", "<variant2>"]
  }},
  "flags": [
    {{"segment_id": <id>, "issue": "<description of problem>"}}
  ]
}}

If everything is consistent and correct, return:
{{"consistency_fixes": [], "entity_map": {{}}, "flags": []}}

REMINDER: Fewer fixes is better. This transcript is already accurate."""


def get_pass1_system_prompt(domain: str | None = None) -> str:
    """Get domain-specific Pass 1 system prompt with safety constraints."""
    domain_key = domain if domain in DOMAIN_PROMPTS else "general"
    domain_prompt = DOMAIN_PROMPTS[domain_key]
    return PASS1_SYSTEM_TEMPLATE.format(domain_prompt=domain_prompt)


def format_segments_for_pass1(
    segments: list,
    window: int = 5,
    low_conf_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    show_word_confidences: bool = True,
) -> str:
    """Format segments with <low_conf> tags around uncertain words.

    Args:
        segments: List of Segment objects with words and confidence
        window: Number of segments before/after for context
        low_conf_threshold: Words below this confidence get <low_conf> tags
        show_word_confidences: Include numeric confidence scores per word

    Returns:
        JSON string with marked_text containing <low_conf> tags
    """
    import json

    if not segments:
        return json.dumps([])

    formatted = []
    for i, seg in enumerate(segments):
        # Get context (read-only)
        context_before = " ".join(
            s.raw_text for s in segments[max(0, i - window):i]
            if s.raw_text
        )
        context_after = " ".join(
            s.raw_text for s in segments[i + 1:min(len(segments), i + window + 1)]
            if s.raw_text
        )

        # Mark low-confidence spans with tags (handle missing/empty words)
        if seg.words:
            marked_text = mark_low_confidence_spans(seg.words, low_conf_threshold)
        else:
            # Fallback to raw text if no word-level data
            marked_text = seg.raw_text or seg.text or ""

        segment_data = {
            "id": seg.id,
            "marked_text": marked_text,
            "overall_confidence": round(seg.confidence, 2),
            "context_before": context_before,
            "context_after": context_after,
        }

        # Include detailed word confidence scores for better decision making
        # This helps Claude understand exactly which words are uncertain
        if show_word_confidences and seg.words:
            segment_data["word_confidences"] = format_word_confidences(
                seg.words, low_conf_threshold
            )

        formatted.append(segment_data)

    return json.dumps(formatted, indent=2)


def format_word_confidences(words: list[WordTiming], threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD) -> list[dict]:
    """Format word-level confidence scores for Claude.

    Returns list of dicts with word, confidence, and whether it's below threshold.
    Only includes words with notable confidence issues (below 0.9) to reduce noise.
    """
    result = []
    for w in words:
        conf = round(w.confidence, 2)
        # Only include words that might need attention (conf < 0.9)
        # High-confidence words are assumed correct
        if conf < 0.9:
            result.append({
                "word": w.word,
                "conf": conf,
                "needs_review": conf < threshold,
            })
    return result


def mark_low_confidence_spans(words: list[WordTiming], threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD) -> str:
    """Mark contiguous low-confidence words with <low_conf> tags.

    Groups adjacent low-confidence words into spans to reduce tag clutter.
    Implements word fragment grouping to handle split proper nouns:
    - Short high-confidence words (1-2 chars) adjacent to low-confidence words
      are included in the span if they likely form a proper noun
    - Example: " Ron" (0.99) + " C" (0.91) + " harrow" (0.37)
      → <low_conf>Ron C harrow</low_conf> (all grouped for correction)

    Args:
        words: List of WordTiming objects with word and confidence
        threshold: Confidence threshold below which words are marked

    Returns:
        Text with <low_conf>...</low_conf> around uncertain spans
    """
    if not words:
        return ""

    # First pass: identify words that should be marked low-confidence
    # This includes both actual low-confidence words AND fragment candidates
    should_mark = []
    for i, w in enumerate(words):
        is_low = w.confidence < threshold

        if is_low:
            should_mark.append(True)
        else:
            # Check if this high-confidence word is a likely fragment
            word_text = w.word.strip()
            is_single_letter = len(word_text) == 1
            is_two_chars = len(word_text) == 2
            is_capitalized = word_text and word_text[0].isupper()

            # Look for adjacent low-confidence words or marked fragments
            has_low_conf_before = i > 0 and words[i - 1].confidence < threshold
            has_low_conf_after = i < len(words) - 1 and words[i + 1].confidence < threshold

            # Fragment detection rules:
            # 1. Single capital letters (like "C" in "C harrow") - likely fragments
            # 2. Two-char capitalized words between/near low-conf regions
            # Exclude common words like "I", "A" that are valid on their own
            is_fragment = False
            if is_single_letter and is_capitalized:
                # Single letters: only mark if NOT "I" or "A" (common valid words)
                if word_text not in ("I", "A"):
                    is_fragment = has_low_conf_before or has_low_conf_after
            elif is_two_chars and is_capitalized:
                # Two-char words: mark if adjacent to low-confidence
                is_fragment = has_low_conf_before or has_low_conf_after

            should_mark.append(is_fragment)

    # Second pass: backward propagation for proper nouns
    # If a capitalized word precedes a marked fragment/low-conf word, include it
    for i in range(len(words) - 1, 0, -1):
        if should_mark[i] and not should_mark[i - 1]:
            prev_word = words[i - 1].word.strip()
            curr_word = words[i].word.strip()
            # Include previous capitalized word if it looks like part of a proper noun
            if (prev_word and prev_word[0].isupper() and
                curr_word and curr_word[0].isupper()):
                should_mark[i - 1] = True

    # Third pass: build spans with grouping
    result = []
    in_low_conf = False
    low_conf_buffer = []

    for i, w in enumerate(words):
        if should_mark[i]:
            if not in_low_conf:
                # Start new low-conf span
                in_low_conf = True
                low_conf_buffer = [w.word]
            else:
                # Continue low-conf span
                low_conf_buffer.append(w.word)
        else:
            if in_low_conf:
                # Close low-conf span
                result.append(f"<low_conf>{' '.join(low_conf_buffer)}</low_conf>")
                in_low_conf = False
                low_conf_buffer = []
            result.append(w.word)

    # Close any remaining span
    if in_low_conf and low_conf_buffer:
        result.append(f"<low_conf>{' '.join(low_conf_buffer)}</low_conf>")

    return " ".join(result)


def extract_high_confidence_text(words: list[WordTiming], threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD) -> list[str]:
    """Extract high-confidence words that should NOT be changed.

    Returns list of (word, position) tuples for diff gating.
    """
    return [
        (w.word, i) for i, w in enumerate(words)
        if w.confidence >= threshold
    ]


def validate_correction(
    original_words: list,
    corrected_text: str,
    threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> tuple[bool, list[str]]:
    """Validate that correction only changed low-confidence regions.

    Implements diff gating: rejects corrections that modify high-confidence text.

    Args:
        original_words: Original WordTiming objects
        corrected_text: The corrected text from Claude
        threshold: Confidence threshold for "protected" words

    Returns:
        (is_valid, violations): Boolean and list of violated words
    """
    # Handle empty/None inputs
    if not original_words or not corrected_text:
        return True, []

    # Get high-confidence words that should be unchanged
    protected_words = []
    for w in original_words:
        # Safely access attributes
        word = getattr(w, 'word', None)
        conf = getattr(w, 'confidence', 0.0)
        if word and conf >= threshold:
            # Strip BOTH whitespace AND punctuation (CrisperWhisper outputs "  word")
            cleaned = word.lower().strip().strip(".,!?;:\"'")
            if cleaned:
                protected_words.append(cleaned)

    # Normalize corrected text for comparison
    corrected_lower = corrected_text.lower()

    # Check each protected word appears in the correction
    violations = []
    for word in protected_words:
        if len(word) < 2:  # Skip punctuation/single chars
            continue
        # Use word boundary matching
        pattern = r'\b' + re.escape(word) + r'\b'
        if not re.search(pattern, corrected_lower):
            violations.append(word)

    # Allow minor variations (punctuation, case) but flag substantial changes
    # If more than 20% of protected words are missing, reject
    if len(protected_words) > 0:
        violation_rate = len(violations) / len(protected_words)
        is_valid = violation_rate < 0.2
    else:
        is_valid = True

    return is_valid, violations


def format_transcript_for_pass2(segments: list) -> str:
    """Format full transcript for Pass 2 prompt."""
    lines = []
    for seg in segments:
        lines.append(f"[{seg.id}] {seg.text}")
    return "\n".join(lines)


def extract_entities_from_pass1(changes_by_segment: dict[int, list]) -> list[str]:
    """Extract unique corrected entities from Pass 1 changes."""
    entities = set()
    for changes in changes_by_segment.values():
        for change in changes:
            # Likely a proper noun if it has capital letters
            if any(c.isupper() for c in change.corrected):
                entities.add(change.corrected)
    return sorted(entities)
