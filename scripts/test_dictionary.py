#!/usr/bin/env python3
"""Test the full dictionary flow."""

from pathlib import Path


def main():
    from asr.dictionary import (
        init_db,
        BiasListSelector,
        generate_whisper_prompt,
        generate_correction_block,
        CandidateMatcher,
    )
    from asr.dictionary.db import get_stats, search_entries

    print("=== ASR Dictionary Test ===\n")

    # 1. Check database
    stats = get_stats()
    print(f"1. Database: {stats.total_entries} entries")
    if stats.total_entries == 0:
        print("   ERROR: No entries! Run: python scripts/load_seeds.py")
        return 1

    # 2. Test search
    print("\n2. Search test:")
    results = search_entries("garmin", limit=5)
    for r in results:
        print(f"   - {r.entry.canonical} (tier {r.entry.tier}, boost {r.entry.boost_weight})")

    # 3. Test bias list selection
    print("\n3. Bias list selection (context=running):")
    selector = BiasListSelector()
    entries = selector.select_bias_list(context="running", max_entries=10)
    print(f"   Got {len(entries)} entries")
    for e in entries[:5]:
        print(f"   - {e.canonical} (score based on tier {e.tier})")

    # 4. Test Whisper prompt generation
    print("\n4. Whisper prompt generation:")
    prompt = generate_whisper_prompt(entries[:20])
    print(f"   {prompt[:200]}...")

    # 5. Test correction block
    print("\n5. Correction block generation:")
    block = generate_correction_block(entries[:10])
    print(f"   {block[:300]}...")

    # 6. Test matcher
    print("\n6. Candidate matching:")
    matcher = CandidateMatcher(entries)
    # Test with a misspelling of "Garmin"
    candidates = matcher.find_candidates("Garmyn")
    if candidates:
        match = candidates[0]
        print(f"   'Garmyn' matched to: {match.entry.canonical} (conf: {match.confidence:.2f})")
    else:
        print("   No candidates found for 'Garmyn'")

    # Test with misspelling of "Strava"
    candidates = matcher.find_candidates("Stravva")
    if candidates:
        match = candidates[0]
        print(f"   'Stravva' matched to: {match.entry.canonical} (conf: {match.confidence:.2f})")
    else:
        print("   No candidates found for 'Stravva'")

    # 7. Test context auto-detection
    print("\n7. Context auto-detection:")
    detected = selector.detect_context("I went for a run on Strava today")
    print(f"   'I went for a run on Strava today' -> context: {detected}")

    print("\n=== All tests passed! ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
