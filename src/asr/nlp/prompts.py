"""Prompts for ASR correction pipeline with safety constraints.

Optimized for CrisperWhisper, a Whisper model fine-tuned for word-level
verbatim transcription. Since CrisperWhisper is already highly accurate,
the correction pipeline is conservative by default.
"""

import re
import html

from asr.models.transcript import WordTiming

# Default low-confidence threshold for marking words that need review
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.7


def _sanitize_prompt_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection attacks.

    Escapes special characters and removes potential prompt manipulation patterns.

    Args:
        text: User-provided input (vocabulary, domain, etc.)

    Returns:
        Sanitized text safe for prompt inclusion
    """
    if not text:
        return ""

    # Remove control characters and non-printable characters
    sanitized = "".join(char for char in text if char.isprintable() or char.isspace())

    # Escape HTML/XML special characters to prevent tag injection
    sanitized = html.escape(sanitized)

    # Remove patterns that could manipulate prompts
    # Block common injection patterns
    injection_patterns = [
        r"system:",
        r"assistant:",
        r"user:",
        r"<\|.*?\|>",  # Special tokens
        r"\[INST\]",   # Instruction markers
        r"\[/INST\]",
    ]

    for pattern in injection_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

    # Limit length to prevent excessive token usage
    MAX_INPUT_LENGTH = 1000
    if len(sanitized) > MAX_INPUT_LENGTH:
        sanitized = sanitized[:MAX_INPUT_LENGTH]

    return sanitized.strip()

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
_GENERAL_PROMPT = """You are an ASR error corrector. Fix OBVIOUS errors only."""
DOMAIN_PROMPTS = {
    "general": CRISPERWHISPER_CONTEXT + _GENERAL_PROMPT + """

CRITICAL RULES:
1. You may ONLY rewrite text inside <low_conf>...</low_conf> tags
2. Outside <low_conf> tags: fix punctuation and capitalization ONLY
3. Never add words that weren't spoken
4. Never paraphrase or formalize language
5. When uncertain, keep the original text
6. Preserve filler words ("um", "uh", "you know")
7. Assume the transcription is probably correct - only fix clear errors""",

    "biography": CRISPERWHISPER_CONTEXT + """You correct ASR for biographical content.

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

    "conversational": CRISPERWHISPER_CONTEXT + """You correct ASR for conversational speech.

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

VOCABULARY HINTS (suggestions, not guarantees):
The vocabulary list contains POSSIBLE terms that MAY appear. They are hints, not facts.
- Only apply a vocabulary term if the ASR output has a NEAR-MISS (phonetically similar word)
- Do NOT force vocabulary terms onto text that doesn't sound similar
- Example: "Ron C harrow" → "Ron Chernow" ✓ (sounds similar)
- Example: "chair" → "Chernow" ✗ (no phonetic match)

Edit each segment independently. Do not reorder or merge sentences across segments.

SEGMENT BOUNDARY RULES (CRITICAL FOR TIMING ALIGNMENT):
- Each segment has a fixed ID, start time, and end time that MUST NOT CHANGE
- Do NOT merge content from one segment into another
- Do NOT split a segment into multiple segments
- Do NOT reorder sentences across segment boundaries
- Edit ONLY the text content within each segment's boundaries
- The segment structure is immutable - only the text inside can change"""


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
      "corrected": "<COMPLETE SEGMENT TEXT with corrections, <low_conf> tags removed>",
      "changes": [
        {{"original": "<word>", "corrected": "<word>", "reason": "<reason>"}}
      ]
    }}
  ]
}}

CRITICAL: The "corrected" field must contain the FULL segment text, not just the changed words.
Example: If input is "I have a <low_conf>Mac book</low_conf> computer", output should be:
  "corrected": "I have a MacBook computer"  (FULL text, not just "MacBook")

IMPORTANT:
- Only include segments where you made changes
- Remove all <low_conf> tags from the output
- Changes should only affect text that was inside <low_conf> tags
- If no changes needed, return {{"corrected_segments": []}}
- FEWER CHANGES IS BETTER - this transcription is already accurate
- Do NOT correct things that "sound wrong" but might be what was said

SEGMENT RULES:
- Segment IDs are immutable - return the same IDs you received
- Do NOT merge or split segments - each segment maps 1:1 to input
- Corrections apply WITHIN segments only, never ACROSS them"""


PASS2_SYSTEM = """You are reviewing a CrisperWhisper transcript for ENTITY CONSISTENCY ONLY.

SOURCE: CrisperWhisper (verbatim-optimized Whisper model)
The transcription has ALREADY been corrected in Pass 1. Your ONLY job now is:

1. Find entities (proper nouns, names, terms) spelled DIFFERENTLY in different places
2. Normalize them to a single canonical spelling

CONSTRAINTS:
- Do NOT rewrite or rephrase any text
- Do NOT fix grammar, punctuation, or capitalization (already done in Pass 1)
- ONLY fix entity spelling inconsistencies (e.g., "MacBook Air" vs "Macbook air")
- If an entity appears only once, do NOT change it
- Fewer fixes is ALWAYS better - only fix clear inconsistencies
- Do NOT introduce new content or words"""


PASS2_USER = """ENTITY CONSISTENCY CHECK

Source: CrisperWhisper (already corrected in Pass 1)
Domain: {domain}
Known vocabulary: {vocabulary}

FULL TRANSCRIPT:
{transcript_text}

ENTITIES FROM PASS 1:
{entities}

YOUR ONLY TASK:
Find entities spelled differently in different places and normalize them.

Example:
- Segment 5: "Macbook air"
- Segment 12: "MacBook Air"
- Segment 23: "macbook Air"
→ Normalize all to "MacBook Air" (most common or vocabulary-matching form)

Return a JSON object:
{{
  "consistency_fixes": [
    {{"segment_id": <id>, "original": "<text>", "corrected": "<text>", "reason": "consistency"}}
  ],
  "entity_map": {{
    "<canonical_spelling>": ["<variant1>", "<variant2>"]
  }},
  "flags": []
}}

If all entities are already consistent, return:
{{"consistency_fixes": [], "entity_map": {{}}, "flags": []}}

REMINDER: You are ONLY checking entity consistency. Do NOT fix anything else."""


# Final kicker pass - uses extended thinking for deep analysis
KICKER_SYSTEM = """You are performing a FINAL REVIEW of a CrisperWhisper transcript.

This transcript has ALREADY been corrected by fast passes. Your role is to use
deep reasoning to catch any subtle issues that were missed.

USE YOUR THINKING to:
1. Read through the entire transcript carefully
2. Identify any remaining inconsistencies or errors
3. Consider context across the full document
4. Make ONLY high-confidence corrections

CONSTRAINTS:
- This transcript is ALREADY highly accurate after previous corrections
- Do NOT introduce new content or words
- Do NOT make speculative corrections
- Only fix things you are HIGHLY confident about
- Prefer NO CHANGES over uncertain changes
- FEWER CHANGES IS ALWAYS BETTER at this stage"""


KICKER_USER = """FINAL REVIEW PASS

Source: CrisperWhisper (verbatim-optimized, already corrected in previous passes)
Domain: {domain}
Known vocabulary: {vocabulary}

FULL TRANSCRIPT (after previous corrections):
{transcript_text}

INSTRUCTIONS:
Use your thinking to carefully review this transcript. Look for:
1. Any remaining entity spelling inconsistencies
2. Obvious errors that previous passes might have missed
3. High-confidence fixes only

Return a JSON object:
{{
  "final_fixes": [
    {{"segment_id": <id>, "original": "<text>", "corrected": "<text>", "reason": "<reason>"}}
  ],
  "quality_assessment": "<brief overall assessment>"
}}

If the transcript is good, return:
{{"final_fixes": [], "quality_assessment": "Transcript is accurate and consistent."}}

REMEMBER: This is the FINAL pass. Only make changes you are HIGHLY confident about."""


def get_pass1_system_prompt(domain: str | None = None) -> str:
    """Get domain-specific Pass 1 system prompt with safety constraints.

    Args:
        domain: Domain name (e.g., "biography", "technical", "medical")

    Returns:
        Formatted system prompt with domain-specific rules
    """
    # Validate domain to prevent injection
    if domain:
        # Only allow alphanumeric and underscore in domain names
        if not re.match(r'^[a-zA-Z0-9_]+$', domain):
            domain = "general"  # Fallback to safe default

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

    # Validate parameters
    if window < 0:
        window = 0
    if window > 100:  # Cap to prevent memory issues
        window = 100

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

    # Compact JSON - no indent means fewer tokens for Haiku
    json_output = json.dumps(formatted, separators=(",", ":"))

    # Token count estimation and warning for very large transcripts
    # Rough estimate: 1 token ≈ 4 characters for JSON
    estimated_tokens = len(json_output) // 4
    MAX_RECOMMENDED_TOKENS = 50000  # Conservative limit for prompt context

    if estimated_tokens > MAX_RECOMMENDED_TOKENS:
        import sys
        print(
            f"Warning: Large transcript detected (~{estimated_tokens} tokens). "
            f"Consider processing in smaller batches to avoid API limits.",
            file=sys.stderr,
        )

    return json_output


def format_word_confidences(
    words: list[WordTiming], threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
) -> list[dict]:
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


def mark_low_confidence_spans(
    words: list[WordTiming], threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
) -> str:
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

    # Handle empty result edge case
    if not result:
        return ""

    return " ".join(result)


def extract_high_confidence_text(
    words: list[WordTiming], threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
) -> list[str]:
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
    # Guard against division by zero when all words are filtered out
    if len(protected_words) > 0:
        violation_rate = len(violations) / len(protected_words)
        is_valid = violation_rate < 0.2
    else:
        # No protected words means nothing to validate
        is_valid = True

    return is_valid, violations


def format_transcript_for_pass2(segments: list) -> str:
    """Format full transcript for Pass 2 prompt.

    Args:
        segments: List of transcript segments

    Returns:
        Formatted transcript string

    Raises:
        ValueError: If segments list is too large (memory protection)
    """
    # Memory protection: limit total segments to prevent OOM
    MAX_SEGMENTS = 10000  # ~100 hours of audio at typical segmentation
    if len(segments) > MAX_SEGMENTS:
        raise ValueError(
            f"Transcript too large: {len(segments)} segments exceeds maximum of {MAX_SEGMENTS}. "
            "Consider processing in smaller batches."
        )

    lines = []
    for seg in segments:
        # Safely handle missing text attribute
        text = getattr(seg, 'text', '')
        seg_id = getattr(seg, 'id', 0)
        lines.append(f"[{seg_id}] {text}")
    return "\n".join(lines)


def validate_phonetic_anchoring(
    original: str,
    corrected: str,
    dictionary_terms: set[str] | None = None,
    min_similarity: float = 0.5,
) -> tuple[bool, str | None]:
    """Validate correction has phonetic anchoring in original text.

    When a correction matches a dictionary term, verify that the original
    ASR output contains a phonetically similar word ("near-miss").

    This prevents hallucinations where Claude forces dictionary terms onto
    unrelated words (e.g., "chair" → "Chernow").

    Args:
        original: The original text before correction
        corrected: The corrected text
        dictionary_terms: Set of dictionary terms (lowercase) to check against.
            If None, skips dictionary validation.
        min_similarity: Minimum phonetic similarity score (0.0-1.0)

    Returns:
        Tuple of (is_valid, reason): is_valid=True if anchoring passes,
        reason explains why validation failed if is_valid=False.
    """
    if not original or not corrected:
        return True, None

    # Only validate corrections that match dictionary terms
    if dictionary_terms is None:
        return True, None

    corrected_lower = corrected.lower().strip()
    if corrected_lower not in dictionary_terms:
        return True, None  # Not a dictionary term, skip validation

    # Check phonetic similarity between original and corrected
    try:
        from rapidfuzz.distance import Levenshtein
        from metaphone import doublemetaphone

        original_words = original.lower().split()
        corrected_words = corrected_lower.split()

        # Get metaphone codes for corrected term
        corrected_meta = []
        for word in corrected_words:
            primary, alternate = doublemetaphone(word)
            if primary:
                corrected_meta.append(primary)

        if not corrected_meta:
            return True, None  # Can't compute metaphone, allow

        # Check if any original word has phonetic match
        for orig_word in original_words:
            orig_word = orig_word.strip(".,!?;:\"'")
            if not orig_word:
                continue

            # Method 1: Edit distance (Levenshtein)
            for corr_word in corrected_words:
                # Normalize by length of longer word
                max_len = max(len(orig_word), len(corr_word))
                # Guard against division by zero for empty strings
                if max_len == 0:
                    continue
                distance = Levenshtein.distance(orig_word, corr_word)
                similarity = 1 - (distance / max_len)
                if similarity >= min_similarity:
                    return True, None  # Found edit distance match

            # Method 2: Phonetic match (Double Metaphone)
            orig_primary, orig_alternate = doublemetaphone(orig_word)

            for corr_meta in corrected_meta:
                # Primary-primary match (strongest)
                if orig_primary and orig_primary == corr_meta:
                    return True, None
                # Primary-alternate or alternate-primary (weaker but valid)
                if orig_alternate and orig_alternate == corr_meta:
                    return True, None

        # No phonetic anchor found
        return False, f"No phonetic match: '{original}' → '{corrected}'"

    except ImportError:
        # Dependencies not available, allow the correction
        return True, None
    except Exception:
        # Error in validation, allow the correction
        return True, None


def sanitize_vocabulary(vocabulary: list[str]) -> list[str]:
    """Sanitize vocabulary list to prevent prompt injection.

    Args:
        vocabulary: List of vocabulary terms from user input

    Returns:
        List of sanitized vocabulary terms safe for prompt inclusion
    """
    if not vocabulary:
        return []

    sanitized = []
    for term in vocabulary:
        if not term or not isinstance(term, str):
            continue

        clean_term = _sanitize_prompt_input(term)
        if clean_term:  # Only include non-empty sanitized terms
            sanitized.append(clean_term)

    # Limit total vocabulary size to prevent token overflow
    MAX_VOCAB_TERMS = 500
    if len(sanitized) > MAX_VOCAB_TERMS:
        sanitized = sanitized[:MAX_VOCAB_TERMS]

    return sanitized


def extract_entities_from_pass1(changes_by_segment: dict[int, list]) -> list[str]:
    """Extract unique corrected entities from Pass 1 changes.

    Args:
        changes_by_segment: Dict mapping segment ID to list of CorrectionChange objects

    Returns:
        Sorted list of sanitized entity strings
    """
    entities = set()
    for changes in changes_by_segment.values():
        for change in changes:
            # Safely access corrected attribute
            corrected = getattr(change, 'corrected', '')
            if not corrected:
                continue

            # Likely a proper noun if it has capital letters
            if any(c.isupper() for c in corrected):
                # Sanitize to prevent prompt injection via entities
                sanitized_entity = _sanitize_prompt_input(corrected)
                if sanitized_entity:  # Only add if sanitization didn't remove everything
                    entities.add(sanitized_entity)

    return sorted(entities)
