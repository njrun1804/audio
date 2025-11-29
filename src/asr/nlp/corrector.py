"""ASR error correction using Claude with two-pass pipeline."""

import json
import os
import sys
import time
import random
from typing import Literal

from pydantic import ValidationError

from asr.logging import get_logger
from asr.models.transcript import CorrectionInfo, Segment
from asr.nlp.models import (
    CorrectionChange,
    CorrectionConfig,
    CorrectionResult,
    KickerResult,
    Pass1Result,
    Pass2Result,
)
from asr.nlp.prompts import (
    KICKER_SYSTEM,
    KICKER_USER,
    PASS1_USER,
    PASS2_SYSTEM,
    PASS2_USER,
    extract_entities_from_pass1,
    format_segments_for_pass1,
    format_transcript_for_pass2,
    get_pass1_system_prompt,
    validate_correction,
    validate_phonetic_anchoring,
)

# API retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
API_TIMEOUT = 60.0  # seconds

# Duration-based batching: target ~30s of audio per batch for even context
TARGET_BATCH_DURATION = 30.0  # seconds of audio per batch
MIN_BATCH_SEGMENTS = 3  # Don't create tiny batches
MAX_BATCH_SEGMENTS = 10  # Cap to avoid token limits


class Corrector:
    """Two-pass ASR error correction pipeline."""

    def __init__(
        self,
        provider: Literal["local", "claude"] = "claude",
        config: CorrectionConfig | None = None,
    ):
        self.provider = provider
        self.config = config or CorrectionConfig()
        self._client = None
        self.accumulated_entities: dict[str, list[str]] = {}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup client."""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection.

        Provides safety net if user forgets to use context manager or call close().
        """
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during garbage collection

    def close(self):
        """Close the Anthropic client and cleanup resources."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client with validation.

        The API key is automatically loaded from:
        1. .env file in project root (auto-loaded on package import)
        2. ANTHROPIC_API_KEY environment variable
        """
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not found. "
                    "Set it in .env file or as environment variable."
                )
            from anthropic import Anthropic
            self._client = Anthropic(timeout=API_TIMEOUT)
        return self._client

    def _call_api_with_retry(
        self,
        model: str,
        system: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        use_thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> tuple[str | None, str | None]:
        """Make API call with retry logic for transient failures.

        Args:
            model: Claude model to use
            system: System prompt
            user_content: User message content
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0.0 = deterministic, reduces hallucinations)
            use_thinking: Enable extended thinking for complex reasoning
            thinking_budget: Token budget for thinking (uses config default if None)

        Returns:
            (response_text, error_message): Either the response text or an error message
        """
        from anthropic import APIError, APIConnectionError, RateLimitError, APIStatusError

        # Use config values if not specified
        if temperature is None:
            temperature = self.config.temperature
        if use_thinking is None:
            use_thinking = self.config.use_extended_thinking
        if thinking_budget is None:
            thinking_budget = self.config.thinking_budget_tokens

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                # When using extended thinking, max_tokens must exceed budget_tokens
                thinking_budget = max(1024, thinking_budget)
                effective_max_tokens = max_tokens
                if use_thinking:
                    # max_tokens must be > thinking budget, add buffer for response
                    effective_max_tokens = thinking_budget + 8192

                # Build API call kwargs
                api_kwargs = {
                    "model": model,
                    "max_tokens": effective_max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user_content}],
                }

                # Extended thinking: let Claude reason before responding
                # This helps with ambiguous corrections and reduces hallucinations
                if use_thinking:
                    api_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }
                    # Extended thinking REQUIRES temperature=1.0 (API constraint)
                    api_kwargs["temperature"] = 1.0
                else:
                    api_kwargs["temperature"] = temperature

                response = self.client.messages.create(**api_kwargs)

                # Validate response structure
                if not response.content:
                    return None, "Empty response from Claude API"

                if not isinstance(response.content, list) or len(response.content) == 0:
                    return None, "No content blocks in API response"

                # Extract text from response, handling extended thinking
                # With thinking enabled, response has thinking blocks + text blocks
                text_content = None
                for block in response.content:
                    if hasattr(block, 'type') and block.type == 'text':
                        text_content = block.text
                        break
                    elif hasattr(block, 'text') and not hasattr(block, 'type'):
                        # Fallback for older response format
                        text_content = block.text
                        break

                if text_content is None:
                    return None, "No text content in API response"

                return text_content, None

            except RateLimitError as e:
                last_error = f"Rate limited: {e}"
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited, retrying in {wait_time:.1f}s...", file=sys.stderr)
                    time.sleep(wait_time)

            except APIConnectionError as e:
                last_error = f"Connection error: {e}"
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                    print(f"Connection error, retrying in {wait_time:.1f}s...", file=sys.stderr)
                    time.sleep(wait_time)

            except APIStatusError as e:
                # Non-retryable errors - SECURITY: Don't expose API key details
                if e.status_code == 401:
                    return None, "Authentication failed. Check your .env file or ANTHROPIC_API_KEY environment variable."
                elif e.status_code == 400:
                    return None, f"Invalid request: {e.message}"
                else:
                    last_error = f"API error ({e.status_code}): {e.message}"

            except APIError as e:
                last_error = f"API error: {e}"

            except Exception as e:
                # Unexpected errors - don't retry
                return None, f"Unexpected error: {type(e).__name__}: {e}"

        return None, f"Failed after {MAX_RETRIES} retries: {last_error}"

    def correct(
        self,
        segments: list[Segment],
        vocabulary: list[str] | None = None,
        domain: str | None = None,
        dictionary_context: str | None = None,
        dictionary_entries_limit: int = 100,
    ) -> list[Segment]:
        """Apply corrections to segments using configured pipeline.

        Args:
            segments: List of transcript segments to correct
            vocabulary: Additional vocabulary terms (trusted spellings)
            domain: Domain context (e.g., "biography", "tech")
            dictionary_context: Dictionary context for loading entries
            dictionary_entries_limit: Max dictionary entries to include in prompt

        Returns:
            List of corrected segments
        """
        # Reset entity accumulator for this file
        self.accumulated_entities = {}

        if self.provider == "local":
            return self._correct_local(segments)

        # Merge vocabulary from config and args
        all_vocab = list(self.config.vocabulary)
        if vocabulary:
            all_vocab.extend(vocabulary)

        # Use domain from args or config
        effective_domain = domain or self.config.domain or "general"

        # Load dictionary entries for correction
        dictionary_entries = []
        dictionary_block = ""
        if dictionary_context:
            try:
                from asr.dictionary import (
                    BiasListSelector,
                    generate_correction_block,
                )

                # BiasListSelector uses SQLite directly - no DictionaryManager needed
                selector = BiasListSelector()
                dictionary_entries = selector.select_bias_list(
                    context=dictionary_context,
                    max_entries=dictionary_entries_limit,
                )

                if dictionary_entries:
                    dictionary_block = generate_correction_block(
                        dictionary_entries, max_entries=dictionary_entries_limit
                    )
            except Exception:
                # Dictionary system is optional
                pass

        # Run the pipeline
        result = CorrectionResult(passes_completed=0, total_changes=0)

        # Pass 1: Correction
        segments, pass1_changes = self._run_pass1(
            segments, all_vocab, effective_domain, dictionary_block, dictionary_entries
        )
        result.passes_completed = 1
        result.changes_by_segment = pass1_changes
        result.total_changes = sum(len(c) for c in pass1_changes.values())

        # Log pass 1 completion
        logger = get_logger()
        if logger:
            logger.log_pass_complete(
                pass_number=1,
                model=self.config.model,
                segments_modified=len(pass1_changes),
            )

        # Pass 2: Consistency (if configured and pass 1 made changes)
        # Skip pass 2 if pass 1 found nothing - consistency check is pointless
        if self.config.passes >= 2 and pass1_changes:
            segments, pass2_result = self._run_pass2(
                segments, all_vocab, effective_domain, pass1_changes
            )
            result.passes_completed = 2
            result.entity_map = pass2_result.entity_map
            result.flags = pass2_result.flags

            # Apply consistency fixes
            for fix in pass2_result.consistency_fixes:
                if fix.segment_id not in result.changes_by_segment:
                    result.changes_by_segment[fix.segment_id] = []
                result.changes_by_segment[fix.segment_id].append(
                    CorrectionChange(
                        original=fix.original,
                        corrected=fix.corrected,
                        reason=f"consistency: {fix.reason}",
                    )
                )
                result.total_changes += 1

            # Log pass 2 completion
            logger = get_logger()
            if logger:
                logger.log_pass_complete(
                    pass_number=2,
                    model=self.config.model,
                    segments_modified=len(pass2_result.consistency_fixes),
                )

        # Final kicker pass: Sonnet/Opus with thinking (if enabled)
        if self.config.use_kicker:
            segments, kicker_result = self._run_kicker(
                segments, all_vocab, effective_domain
            )

            # Apply kicker fixes to result
            for fix in kicker_result.final_fixes:
                if fix.segment_id not in result.changes_by_segment:
                    result.changes_by_segment[fix.segment_id] = []
                result.changes_by_segment[fix.segment_id].append(
                    CorrectionChange(
                        original=fix.original,
                        corrected=fix.corrected,
                        reason=f"kicker: {fix.reason}",
                    )
                )
                result.total_changes += 1

            # Log kicker completion
            logger = get_logger()
            if logger:
                logger.log_pass_complete(
                    pass_number=3,  # Kicker is pass 3
                    model=self.config.kicker_model,
                    segments_modified=len(kicker_result.final_fixes),
                )

        return segments

    def _create_duration_batches(self, segments: list[Segment]) -> list[list[Segment]]:
        """Create batches targeting ~30s of audio each for even context windows.

        This replaces fixed segment-count batching which creates uneven contexts
        because segment durations vary from 2s to 30s.
        """
        if not segments:
            return []

        batches = []
        current_batch = []
        current_duration = 0.0

        for seg in segments:
            seg_duration = seg.end - seg.start

            # Check if adding this segment would exceed target duration
            # Also enforce min/max segment counts
            would_exceed_duration = (current_duration + seg_duration) > TARGET_BATCH_DURATION
            at_max_segments = len(current_batch) >= MAX_BATCH_SEGMENTS

            if current_batch and (would_exceed_duration or at_max_segments):
                # Start new batch if we have enough segments
                if len(current_batch) >= MIN_BATCH_SEGMENTS:
                    batches.append(current_batch)
                    current_batch = [seg]
                    current_duration = seg_duration
                else:
                    # Keep adding to reach minimum
                    current_batch.append(seg)
                    current_duration += seg_duration
            else:
                current_batch.append(seg)
                current_duration += seg_duration

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def _run_pass1(
        self,
        segments: list[Segment],
        vocabulary: list[str],
        domain: str,
        dictionary_block: str = "",
        dictionary_entries: list | None = None,
    ) -> tuple[list[Segment], dict[int, list[CorrectionChange]]]:
        """Run Pass 1: Intelligent correction."""
        changes_by_segment: dict[int, list[CorrectionChange]] = {}
        corrected_texts: dict[int, str] = {}  # Track corrected text by segment ID
        dictionary_entries = dictionary_entries or []

        # Build dictionary terms set for phonetic anchoring validation
        dict_terms: set[str] | None = None
        if dictionary_entries:
            dict_terms = set()
            for entry in dictionary_entries:
                dict_terms.add(entry.canonical.lower())
                if entry.display:
                    dict_terms.add(entry.display.lower())
                for alias in entry.aliases:
                    dict_terms.add(alias.alias.lower())

        # Process in duration-based batches
        batches = self._create_duration_batches(segments)
        for batch in batches:
            batch_changes, batch_texts = self._correct_batch(
                batch, vocabulary, domain, dictionary_block, dict_terms,
                entity_context=self.accumulated_entities,
            )
            changes_by_segment.update(batch_changes)
            corrected_texts.update(batch_texts)

            # Accumulate entities from corrections for consistency across batches
            for seg_id, changes in batch_changes.items():
                for change in changes:
                    # Track proper nouns (capitalized words that were corrected)
                    corrected = change.corrected.strip()
                    original = change.original.strip()
                    if corrected and corrected[0].isupper() and corrected != original:
                        canonical = corrected
                        if canonical not in self.accumulated_entities:
                            self.accumulated_entities[canonical] = []
                        existing = [v.lower() for v in self.accumulated_entities[canonical]]
                        if original.lower() not in existing:
                            self.accumulated_entities[canonical].append(original)
                        # Cap variants at 10 per canonical
                        self.accumulated_entities[canonical] = self.accumulated_entities[canonical][:10]

            # Track dictionary usage for corrections that match entries
            if dictionary_entries and batch_changes:
                self._track_dictionary_usage(batch_changes, dictionary_entries)

        # Build corrected segments with updated text AND corrections metadata
        corrected_segments = []
        for seg in segments:
            if seg.id in changes_by_segment:
                changes = changes_by_segment[seg.id]
                corrected_seg = seg.model_copy(
                    update={
                        "text": corrected_texts.get(seg.id, seg.text),
                        "corrections": CorrectionInfo(
                            applied=True,
                            source="claude",
                            changes=[f"{c.original} -> {c.corrected}" for c in changes],
                        ),
                    }
                )
                corrected_segments.append(corrected_seg)
            else:
                corrected_segments.append(seg)

        return corrected_segments, changes_by_segment

    def _correct_batch(
        self,
        batch: list[Segment],
        vocabulary: list[str],
        domain: str,
        dictionary_block: str = "",
        dictionary_terms: set[str] | None = None,
        entity_context: dict[str, list[str]] | None = None,
    ) -> tuple[dict[int, list[CorrectionChange]], dict[int, str]]:
        """Correct a batch of segments using Claude with safety constraints.

        Uses <low_conf> tags to mark uncertain regions and diff gating to reject
        corrections that modify high-confidence text. Also validates phonetic
        anchoring for dictionary term corrections.
        """
        # Format segments with <low_conf> tags around uncertain words
        # Include word-level confidence scores for better decision making
        segments_json = format_segments_for_pass1(
            batch,
            window=self.config.context_window,
            low_conf_threshold=self.config.low_confidence_threshold,
            show_word_confidences=self.config.show_word_confidences,
        )
        vocab_str = ", ".join(vocabulary) if vocabulary else "None provided"

        # Format dictionary block with leading newline if present
        dict_block_formatted = f"\n{dictionary_block}\n" if dictionary_block else ""

        # Build entity context block if we have accumulated entities
        entity_context_block = ""
        if entity_context:
            entity_lines = []
            for canonical, variants in list(entity_context.items())[:20]:  # Limit to 20 entities
                if variants:
                    entity_lines.append(f"  - \"{canonical}\" (may appear as: {', '.join(variants[:5])})")
                else:
                    entity_lines.append(f"  - \"{canonical}\"")
            if entity_lines:
                entity_context_block = "\nKnown entities from earlier segments:\n" + "\n".join(entity_lines) + "\n"

        user_prompt = PASS1_USER.format(
            domain=domain,
            vocabulary=vocab_str,
            aggressiveness=self.config.aggressiveness,
            dictionary_block=dict_block_formatted,
            segments_json=segments_json,
        )

        # Prepend entity context before segments_json in the user prompt
        if entity_context_block:
            # Insert entity context before segments_json
            user_prompt = user_prompt.replace(segments_json, entity_context_block + segments_json)

        # Get domain-specific system prompt with safety rules
        system_prompt = get_pass1_system_prompt(domain)

        # Call API with retry logic
        response_text, error = self._call_api_with_retry(
            model=self.config.model,
            system=system_prompt,
            user_content=user_prompt,
        )

        if error:
            print(f"Warning: Claude API call failed: {error}", file=sys.stderr)
            return {}, {}

        # Parse JSON response
        changes_by_segment: dict[int, list[CorrectionChange]] = {}
        corrected_texts: dict[int, str] = {}

        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = self._extract_json_from_response(response_text)
            result = Pass1Result.model_validate_json(json_str.strip())

            for seg_correction in result.corrected_segments:
                # Find the segment and apply diff gating
                found_segment = False
                rejected = False

                for seg in batch:
                    if seg.id == seg_correction.id:
                        found_segment = True
                        # DIFF GATING: Validate that high-confidence text wasn't changed
                        if self.config.strict_diff_gating:
                            is_valid, violations = validate_correction(
                                seg.words,
                                seg_correction.corrected,
                                self.config.low_confidence_threshold,
                            )
                            if not is_valid:
                                print(
                                    f"Warning: Rejected correction for segment {seg.id} "
                                    f"- modified protected words: {violations[:5]}",
                                    file=sys.stderr,
                                )
                                # Log rejection
                                logger = get_logger()
                                if logger:
                                    logger.log_correction_rejected(
                                        segment_id=seg.id,
                                        reason="diff_gating",
                                        violations=violations,
                                        details={
                                            "attempted_correction": seg_correction.corrected[:200],
                                            "original_text": seg.text[:200],
                                        },
                                    )
                                # Mark as rejected and break from inner loop
                                rejected = True
                                break

                        # PHONETIC ANCHORING: Validate dictionary term corrections
                        # Only apply if each change has a phonetic "near-miss" in original
                        valid_changes = []
                        for c in seg_correction.changes:
                            is_anchored, reason = validate_phonetic_anchoring(
                                c.original,
                                c.corrected,
                                dictionary_terms=dictionary_terms,
                            )
                            if is_anchored:
                                valid_changes.append(c)
                            else:
                                print(
                                    f"Warning: Rejected change '{c.original}' → '{c.corrected}' "
                                    f"- {reason}",
                                    file=sys.stderr,
                                )
                                # Log rejection
                                logger = get_logger()
                                if logger:
                                    logger.log_correction_rejected(
                                        segment_id=seg.id,
                                        reason="phonetic_anchoring",
                                        violations=[reason] if reason else [],
                                        details={
                                            "original": c.original,
                                            "corrected": c.corrected,
                                        },
                                    )

                        # If all changes were rejected, skip this segment
                        if not valid_changes and seg_correction.changes:
                            rejected = True
                            break

                        # Only apply if diff gating passed or is disabled
                        corrected_texts[seg.id] = seg_correction.corrected
                        changes_by_segment[seg.id] = [
                            CorrectionChange(
                                original=c.original,
                                corrected=c.corrected,
                                reason=c.reason,
                            )
                            for c in valid_changes
                        ]
                        # Log successful correction
                        logger = get_logger()
                        if logger:
                            logger.log_correction_applied(
                                segment_id=seg.id,
                                changes=[
                                    {"original": c.original, "corrected": c.corrected, "reason": c.reason}
                                    for c in seg_correction.changes
                                ],
                                corrected_text=seg_correction.corrected,
                            )
                        break

                if rejected:
                    continue  # Skip to next correction

                if not found_segment:
                    seg_id = seg_correction.id
                    print(f"Warning: Segment ID {seg_id} not found in batch", file=sys.stderr)

        except json.JSONDecodeError as e:
            print(
                f"Warning: Failed to parse correction response (invalid JSON): {e}",
                file=sys.stderr,
            )
        except ValidationError as e:
            print(f"Warning: Response doesn't match expected schema: {e}", file=sys.stderr)
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse correction response: {e}", file=sys.stderr)

        return changes_by_segment, corrected_texts

    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        if not text:
            return "{}"

        json_str = text

        # Try to extract from markdown code blocks
        if "```json" in json_str:
            parts = json_str.split("```json")
            if len(parts) >= 2:
                content = parts[1]
                if "```" in content:
                    json_str = content.split("```")[0]
                else:
                    json_str = content
        elif "```" in json_str:
            parts = json_str.split("```")
            if len(parts) >= 2:
                json_str = parts[1]

        return json_str.strip()

    def _run_pass2(
        self,
        segments: list[Segment],
        vocabulary: list[str],
        domain: str,
        pass1_changes: dict[int, list[CorrectionChange]],
    ) -> tuple[list[Segment], Pass2Result]:
        """Run Pass 2: Consistency check."""
        # Format transcript
        transcript_text = format_transcript_for_pass2(segments)
        entities = extract_entities_from_pass1(pass1_changes)
        vocab_str = ", ".join(vocabulary) if vocabulary else "None provided"
        entities_str = ", ".join(entities) if entities else "None identified"

        user_prompt = PASS2_USER.format(
            domain=domain,
            vocabulary=vocab_str,
            transcript_text=transcript_text,
            entities=entities_str,
        )

        # Call API with retry logic (always Sonnet with thinking)
        response_text, error = self._call_api_with_retry(
            model=self.config.model,
            system=PASS2_SYSTEM,
            user_content=user_prompt,
        )

        if error:
            print(f"Warning: Claude API call failed in pass 2: {error}", file=sys.stderr)
            return segments, Pass2Result()

        try:
            # Extract JSON from response
            json_str = self._extract_json_from_response(response_text)
            result = Pass2Result.model_validate_json(json_str.strip())

            # Apply consistency fixes - PERFORMANCE: Use dict lookup O(1) instead of O(n²)
            fixes_by_segment = {fix.segment_id: fix for fix in result.consistency_fixes}

            corrected_segments = []
            for seg in segments:
                fix_applied = fixes_by_segment.get(seg.id)

                if fix_applied:
                    corrected_seg = seg.model_copy(
                        update={"text": fix_applied.corrected}
                    )
                    corrected_segments.append(corrected_seg)
                else:
                    corrected_segments.append(seg)

            return corrected_segments, result

        except json.JSONDecodeError as e:
            print(
                f"Warning: Failed to parse consistency response (invalid JSON): {e}",
                file=sys.stderr,
            )
            return segments, Pass2Result()
        except ValidationError as e:
            print(f"Warning: Response doesn't match expected schema: {e}", file=sys.stderr)
            return segments, Pass2Result()
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse consistency response: {e}", file=sys.stderr)
            return segments, Pass2Result()

    def _run_kicker(
        self,
        segments: list[Segment],
        vocabulary: list[str],
        domain: str,
    ) -> tuple[list[Segment], KickerResult]:
        """Run final kicker pass: Sonnet/Opus with extended thinking.

        This is the final polish pass that uses deep reasoning to catch
        any subtle issues missed by the fast Haiku passes.
        """
        # Format transcript
        transcript_text = format_transcript_for_pass2(segments)
        vocab_str = ", ".join(vocabulary) if vocabulary else "None provided"

        user_prompt = KICKER_USER.format(
            domain=domain,
            vocabulary=vocab_str,
            transcript_text=transcript_text,
        )

        # Call API with kicker model and THINKING enabled
        response_text, error = self._call_api_with_retry(
            model=self.config.kicker_model,
            system=KICKER_SYSTEM,
            user_content=user_prompt,
            use_thinking=True,  # Enable extended thinking for deep analysis
            thinking_budget=self.config.kicker_thinking_budget,
        )

        if error:
            print(f"Warning: Kicker API call failed: {error}", file=sys.stderr)
            return segments, KickerResult()

        try:
            # Extract JSON from response
            json_str = self._extract_json_from_response(response_text)
            result = KickerResult.model_validate_json(json_str.strip())

            # Apply final fixes
            fixes_by_segment = {fix.segment_id: fix for fix in result.final_fixes}

            corrected_segments = []
            for seg in segments:
                fix_applied = fixes_by_segment.get(seg.id)

                if fix_applied:
                    corrected_seg = seg.model_copy(
                        update={"text": fix_applied.corrected}
                    )
                    corrected_segments.append(corrected_seg)
                else:
                    corrected_segments.append(seg)

            return corrected_segments, result

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse kicker response (invalid JSON): {e}", file=sys.stderr)
            return segments, KickerResult()
        except ValidationError as e:
            print(f"Warning: Kicker response doesn't match expected schema: {e}", file=sys.stderr)
            return segments, KickerResult()
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse kicker response: {e}", file=sys.stderr)
            return segments, KickerResult()

    def extract_learnable_terms(
        self,
        segments: list[Segment],
        changes_by_segment: dict[int, list],
    ) -> tuple[list[str], set[str]]:
        """Extract terms that could be added to vocabulary.

        SAFETY: Returns both the candidate terms AND the original words
        so the caller can verify terms weren't hallucinated.

        Args:
            segments: List of transcript segments to analyze
            changes_by_segment: Dict mapping segment ID to list of corrections

        Returns:
            Tuple of (candidate_terms, original_words) where:
            - candidate_terms: List of proper noun strings to potentially learn
            - original_words: Set of words from original ASR output for verification
        """
        candidates = []
        original_words = set()

        # Collect all original words from segments
        for seg in segments:
            # Safe access: check if words list exists
            if seg.words:
                for word in seg.words:
                    if word.word:
                        original_words.add(word.word)
            # Also add raw text words - safe access with None check
            if seg.raw_text:
                for word in seg.raw_text.split():
                    stripped = word.strip(".,!?;:\"'")
                    if stripped:
                        original_words.add(stripped)

        # Extract candidates from corrections
        for seg_id, changes in changes_by_segment.items():
            for change in changes:
                # Only consider proper nouns (capitalized) or multi-word terms
                corrected = change.corrected.strip()
                if not corrected:
                    continue

                # Check if it's likely a proper noun or technical term
                # Safe access: check length before accessing index
                is_proper_noun = len(corrected) > 0 and corrected[0].isupper()
                is_multi_word = len(corrected.split()) > 1
                reason_suggests_entity = any(
                    kw in change.reason.lower()
                    for kw in ["name", "proper", "entity", "company", "product", "person"]
                )

                if is_proper_noun or is_multi_word or reason_suggests_entity:
                    candidates.append(corrected)

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_candidates.append(c)

        return unique_candidates, original_words

    def _correct_local(self, segments: list[Segment]) -> list[Segment]:
        """Basic local correction (punctuation, capitalization)."""
        corrected = []

        for seg in segments:
            text = seg.raw_text.strip()

            # Skip empty text
            if not text:
                corrected.append(seg)
                continue

            # Basic capitalization - safe access with length check
            if len(text) >= 1 and text[0].islower():
                if len(text) > 1:
                    text = text[0].upper() + text[1:]
                else:
                    text = text[0].upper()

            # Basic punctuation
            if len(text) >= 1 and text[-1] not in ".!?":
                text = text + "."

            changes = []
            if text != seg.raw_text:
                changes.append("basic_punctuation")

            corrected_seg = seg.model_copy(
                update={
                    "text": text,
                    "corrections": CorrectionInfo(
                        applied=bool(changes),
                        source="local" if changes else "none",
                        changes=changes,
                    ),
                }
            )
            corrected.append(corrected_seg)

        return corrected

    def _track_dictionary_usage(
        self,
        changes_by_segment: dict[int, list[CorrectionChange]],
        dictionary_entries: list,
    ) -> None:
        """Track dictionary entry usage when corrections match entries.

        For each correction that matches a dictionary entry:
        - Record usage (increment occurrence_count, update last_seen)
        - If the original text is a consistent misspelling, consider adding as alias

        Args:
            changes_by_segment: Dict mapping segment ID to list of changes
            dictionary_entries: List of EntryWithRelations from the dictionary
        """
        if not dictionary_entries or not changes_by_segment:
            return

        try:
            from asr.dictionary import DictionaryManager

            manager = DictionaryManager()

            # Build lookup of canonical forms and aliases to entry IDs
            form_to_entry: dict[str, str] = {}  # lowercase form -> entry ID
            for entry in dictionary_entries:
                form_to_entry[entry.canonical.lower()] = entry.id
                if entry.display:
                    form_to_entry[entry.display.lower()] = entry.id
                for alias in entry.aliases:
                    form_to_entry[alias.alias.lower()] = entry.id

            # Check each correction
            for seg_id, changes in changes_by_segment.items():
                for change in changes:
                    corrected_lower = change.corrected.lower().strip()

                    # Check if correction matches a dictionary entry
                    if corrected_lower in form_to_entry:
                        entry_id = form_to_entry[corrected_lower]

                        # Record usage
                        manager.record_usage(entry_id)

                        # If original was different and looks like misspelling, track it
                        original_lower = change.original.lower().strip()
                        if original_lower != corrected_lower and original_lower:
                            # Use maybe_add_alias to track potential misspellings
                            # It requires 3+ occurrences before promoting to real alias
                            manager.maybe_add_alias(
                                entry_id, change.original, min_occurrences=3
                            )

        except Exception:
            # Dictionary tracking is optional - don't fail the correction
            pass
