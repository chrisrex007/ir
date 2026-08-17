"""The Porter stemming algorithm (M.F. Porter, 1980).

Written from the algorithm definition published at
https://tartarus.org/martin/PorterStemmer/def.txt

The stemmer works on a single lowercase alphabetic word and applies the five
steps of the algorithm in order, each step rewriting the suffix of the word.
"""

VOWELS = "aeiou"


def _is_consonant(word, i):
    """A letter is a consonant unless it is a vowel, or a 'y' after a consonant."""
    ch = word[i]
    if ch in VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem):
    """Count m, the number of VC sequences in the form [C](VC)^m[V]."""
    n = len(stem)
    i = 0
    while i < n and _is_consonant(stem, i):
        i += 1
    m = 0
    while i < n:
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        m += 1
        while i < n and _is_consonant(stem, i):
            i += 1
    return m


def _has_vowel(stem):
    """The *v* condition: the stem contains a vowel."""
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem):
    """The *d condition: the stem ends in a doubled consonant."""
    return (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and _is_consonant(stem, len(stem) - 1)
    )


def _ends_cvc(stem):
    """The *o condition: the stem ends consonant-vowel-consonant, last not w/x/y."""
    n = len(stem)
    if n < 3:
        return False
    if not (
        _is_consonant(stem, n - 3)
        and not _is_consonant(stem, n - 2)
        and _is_consonant(stem, n - 1)
    ):
        return False
    return stem[-1] not in "wxy"


def _replace(word, suffix, replacement, min_measure=None):
    """Rewrite `suffix` as `replacement` if the stem satisfies m > min_measure.

    Returns the new word, or None when the rule does not apply.
    """
    if not word.endswith(suffix):
        return None
    stem = word[: len(word) - len(suffix)]
    if min_measure is not None and _measure(stem) <= min_measure:
        return None
    return stem + replacement


def _step1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word):
    if word.endswith("eed"):
        stem = word[:-3]
        return word[:-1] if _measure(stem) > 0 else word

    stem = None
    if word.endswith("ed") and _has_vowel(word[:-2]):
        stem = word[:-2]
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        stem = word[:-3]
    if stem is None:
        return word

    # The clean-up rules that follow a successful -ed / -ing removal.
    if stem.endswith(("at", "bl", "iz")):
        return stem + "e"
    if _ends_double_consonant(stem) and stem[-1] not in "lsz":
        return stem[:-1]
    if _measure(stem) == 1 and _ends_cvc(stem):
        return stem + "e"
    return stem


def _step1c(word):
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


_STEP2_RULES = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]

_STEP3_RULES = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]

_STEP4_SUFFIXES = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
    "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def _step2(word):
    for suffix, replacement in _STEP2_RULES:
        new = _replace(word, suffix, replacement, min_measure=0)
        if new is not None:
            return new
    return word


def _step3(word):
    for suffix, replacement in _STEP3_RULES:
        new = _replace(word, suffix, replacement, min_measure=0)
        if new is not None:
            return new
    return word


def _step4(word):
    # -ion is only removed after an s or a t.
    if word.endswith("ion"):
        stem = word[:-3]
        if stem.endswith(("s", "t")) and _measure(stem) > 1:
            return stem
    for suffix in _STEP4_SUFFIXES:
        new = _replace(word, suffix, "", min_measure=1)
        if new is not None:
            return new
    return word


def _step5a(word):
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            return stem
    return word


def _step5b(word):
    if word.endswith("l") and _ends_double_consonant(word) and _measure(word) > 1:
        return word[:-1]
    return word


def stem(word):
    """Return the Porter stem of a single lowercase word."""
    if len(word) <= 2:
        return word
    for step in (_step1a, _step1b, _step1c, _step2, _step3, _step4, _step5a, _step5b):
        word = step(word)
    return word
