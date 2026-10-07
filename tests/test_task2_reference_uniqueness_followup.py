from collections import Counter

import numpy as np
import pytest

from scripts.task2_reference_uniqueness_followup import circular_unique_starts


def naive(sequence: bytes, k: int):
    sequence = sequence.upper()
    circular = sequence + sequence[:k - 1]
    translate = bytes.maketrans(b"ACGT", b"TGCA")
    words = [circular[i:i+k] for i in range(len(sequence))]
    valid = [all(base in b"ACGT" for base in word) for word in words]
    keys = [min(word, word.translate(translate)[::-1]) for word in words]
    counts = Counter(key for key, ok in zip(keys, valid) if ok)
    return np.array([ok and counts[key] == 1 for key, ok in zip(keys, valid)]), np.array(valid)


@pytest.mark.parametrize("sequence,k", [(b"AACGTTGCA", 3), (b"AAAACCCCGGGGTTTT", 7),
                                      (b"ACNTGA", 3), (b"acgtaagc", 2), (b"ACGT", 4)])
def test_exact_circular_counts_match_naive_both_strands(sequence, k):
    actual = circular_unique_starts(sequence, k)
    expected = naive(sequence, k)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


def test_packed_codes_longer_than_machine_word_are_exact():
    rng = np.random.default_rng(3)
    sequence = bytes(rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), size=160))
    for k in (31, 51, 101):
        actual, valid = circular_unique_starts(sequence, k)
        expected, expected_valid = naive(sequence, k)
        assert np.array_equal(actual, expected) and np.array_equal(valid, expected_valid)


def test_repetitive_genome_and_invalid_k():
    assert not circular_unique_starts(b"ACACACAC", 3)[0].any()
    for k in (0, 9, True, 2.5):
        with pytest.raises(ValueError):
            circular_unique_starts(b"ACGTACGT", k)
