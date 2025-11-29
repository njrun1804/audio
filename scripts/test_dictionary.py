#!/usr/bin/env python3
"""Test the full dictionary flow."""

import sys


def main():
    try:
        from asr.dictionary import (
            BiasListSelector,
            generate_whisper_prompt,
            generate_correction_block,
            CandidateMatcher,
        )
        from asr.dictionary.db import get_stats, search_entries
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}", file=sys.stderr)
        print("Make sure the asr package is installed: pip install -e .", file=sys.stderr)
        return 1

    print("=== ASR Dictionary Test ===\n")

    # 1. Check database
    try:
        stats = get_stats()
    except Exception as e:
        print(f"ERROR: Failed to get database stats: {e}", file=sys.stderr)
        return 1

    print(f"1. Database: {stats.total_entries} entries")
    if stats.total_entries == 0:
        print("   ERROR: No entries! Run: python scripts/load_seeds.py")
        return 1

    # 2. Test search
    print("\n2. Search test:")
    try:
        results = search_entries("garmin", limit=5)
        for r in results:
            print(f"   - {r.entry.canonical} (tier {r.entry.tier}, boost {r.entry.boost_weight})")
    except Exception as e:
        print(f"ERROR: Search failed: {e}", file=sys.stderr)
        return 1

    # 3. Test bias list selection
    print("\n3. Bias list selection (context=running):")
    try:
        selector = BiasListSelector()
        entries = selector.select_bias_list(context="running", max_entries=10)
        print(f"   Got {len(entries)} entries")
        for e in entries[:5]:
            print(f"   - {e.canonical} (score based on tier {e.tier})")
    except Exception as e:
        print(f"ERROR: Bias list selection failed: {e}", file=sys.stderr)
        return 1

    # 4. Test Whisper prompt generation
    print("\n4. Whisper prompt generation:")
    try:
        prompt = generate_whisper_prompt(entries[:20])
        print(f"   {prompt[:200]}...")
    except Exception as e:
        print(f"ERROR: Whisper prompt generation failed: {e}", file=sys.stderr)
        return 1

    # 5. Test correction block
    print("\n5. Correction block generation:")
    try:
        block = generate_correction_block(entries[:10])
        print(f"   {block[:300]}...")
    except Exception as e:
        print(f"ERROR: Correction block generation failed: {e}", file=sys.stderr)
        return 1

    # 6. Test matcher
    print("\n6. Candidate matching:")
    try:
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
            canonical = match.entry.canonical
            print(f"   'Stravva' matched to: {canonical} (conf: {match.confidence:.2f})")
        else:
            print("   No candidates found for 'Stravva'")
    except Exception as e:
        print(f"ERROR: Candidate matching failed: {e}", file=sys.stderr)
        return 1

    # 7. Test context auto-detection
    print("\n7. Context auto-detection:")
    try:
        detected = selector.detect_context("I went for a run on Strava today")
        print(f"   'I went for a run on Strava today' -> context: {detected}")
    except Exception as e:
        print(f"ERROR: Context auto-detection failed: {e}", file=sys.stderr)
        return 1

    print("\n=== All tests passed! ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
