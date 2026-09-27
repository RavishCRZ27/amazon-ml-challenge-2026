"""Dependency-free romanizer for Indic scripts + Latin accent folding.

Uses only unicodedata character NAMES, so one code path covers Devanagari,
Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam.
Deterministic character table from the Python stdlib: no model, no network,
no external data.

  to_latin("आदित्य प्रोडक्ट्स")      -> "adity prodakts"
  to_latin("ಹರಿ ಲಾಜಿಸ್ಟಿಕ್ಸ್")       -> "hari lajistiks"
  to_latin("Léarning Çoncept")        -> "Learning Concept"
"""
import functools
import re
import unicodedata as ud

_INDIC = {"DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA",
          "TAMIL", "TELUGU", "KANNADA", "MALAYALAM"}
# Scripts whose word-final inherent vowel is usually silent (schwa deletion).
_SCHWA_DROP = {"DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI"}

_VOWELS = {  # long/short collapsed on purpose: we match against English spellings
    "A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u",
    "E": "e", "EE": "e", "AI": "ai", "O": "o", "OO": "o", "AU": "au",
    "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li", "VOCALIC LL": "li",
    "CANDRA E": "e", "CANDRA O": "o", "SHORT E": "e", "SHORT O": "o", "CANDRA A": "a",
}
# Consonant = Unicode letter name minus its trailing inherent "A".
_CONS = {
    "K": "k", "KH": "kh", "G": "g", "GH": "gh", "NG": "n",
    "C": "ch", "CH": "chh", "J": "j", "JH": "jh", "NY": "n",
    "TT": "t", "TTH": "th", "DD": "d", "DDH": "dh", "NN": "n",
    "T": "t", "TH": "th", "D": "d", "DH": "dh", "N": "n", "NNN": "n",
    "P": "p", "PH": "ph", "B": "b", "BH": "bh", "M": "m",
    "Y": "y", "YY": "y", "R": "r", "RR": "r", "L": "l", "LL": "l", "LLL": "l",
    "V": "v", "W": "v", "SH": "sh", "SS": "sh", "S": "s", "H": "h",
    "Q": "q", "KHH": "kh", "GHH": "g", "Z": "z", "F": "f",
    "DDDH": "r", "RH": "rh", "TTT": "t", "KHANDA T": "t",
}
_NUKTA = {"k": "q", "kh": "kh", "g": "g", "j": "z", "ph": "f", "d": "r", "dh": "rh"}
_NASAL_LABIAL = re.compile(r"\x00(?=[pbm])")
_NASAL = {"SIGN ANUSVARA", "SIGN CANDRABINDU", "SIGN BINDI", "TIPPI"}
_LIGATURES = {"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ß": "ss", "ø": "o", "Ø": "O",
              "ł": "l", "Ł": "L", "đ": "d", "Đ": "D"}


@functools.lru_cache(maxsize=4096)
def _indic(ch):
    try:
        name = ud.name(ch)
    except ValueError:
        return None, None
    script, _, rest = name.partition(" ")
    return (script, rest) if script in _INDIC else (None, None)


def _romanize_indic(s):
    out, pend, script, last = [], False, None, -1

    def realize():  # inherent vowel is spoken before whatever comes next
        nonlocal pend
        if pend:
            out.append("a")
            pend = False

    for ch in s:
        sc, rest = _indic(ch)
        if sc is None:                      # word boundary / Latin / punctuation
            if pend and script not in _SCHWA_DROP:
                out.append("a")
            pend = False
            out.append(ch)
            continue
        script = sc
        if rest.startswith("LETTER CHILLU "):            # Malayalam dead consonant
            realize(); out.append(_CONS.get(rest[14:], rest[14:].lower())); continue
        if rest.startswith("LETTER "):
            x = rest[7:]
            if x in _VOWELS:                              # independent vowel
                realize(); out.append(_VOWELS[x]); continue
            realize()
            base = x[:-1] if x.endswith("A") else x
            out.append(_CONS.get(base, base.lower())); last = len(out) - 1; pend = True
            continue
        if rest.startswith("VOWEL SIGN "):
            out.append(_VOWELS.get(rest[11:], "")); pend = False; continue
        if rest == "SIGN VIRAMA":
            pend = False; continue
        if rest == "SIGN NUKTA" and last >= 0:
            out[last] = _NUKTA.get(out[last], out[last]); continue
        if rest in _NASAL:
            realize(); out.append("\x00"); continue
        if rest == "SIGN VISARGA":
            realize(); out.append("h"); continue
        if rest.startswith("DIGIT "):
            realize(); out.append(str(ud.digit(ch))); continue
        if rest in ("DANDA", "DOUBLE DANDA"):
            realize(); out.append(" "); continue
        # ADDAK, AVAGRAHA, length marks, etc.: no Latin equivalent, skip
    if pend and script not in _SCHWA_DROP:
        out.append("a")
    # nasal sign is written M before labials in English (मुंबई -> mumbai), else N
    return _NASAL_LABIAL.sub("m", "".join(out)).replace("\x00", "n")


_NON_ASCII = re.compile(r"[^\x00-\x7f]")
# Malayalam spells the English "tt"/"nt" sounds with RRA clusters.
_PRE = {"റ്റ": "ട്ട",   # റ്റ -> ട്ട  (tt)
        "ന്റ": "ന്ട"}   # ന്റ -> ന്ട  (nt)


def to_latin(s):
    """Romanize Indic runs, then fold Latin accents. ASCII input is returned unchanged."""
    if s is None or not _NON_ASCII.search(s):
        return s
    for k, v in _PRE.items():
        s = s.replace(k, v)
    s = _romanize_indic(s)
    s = "".join(_LIGATURES.get(c, c) for c in s)
    s = ud.normalize("NFKD", s)
    return "".join(c for c in s if not ud.combining(c))


def script_class(s):
    """'ascii' | 'latin_accented' | 'indic' | 'other' — a feature for the matcher."""
    if s is None or not _NON_ASCII.search(s):
        return "ascii"
    kinds = set()
    for c in s:
        if ord(c) < 128:
            continue
        sc, _ = _indic(c)
        if sc:
            kinds.add("indic")
        elif ud.name(c, "").startswith("LATIN"):
            kinds.add("latin")
        elif not ud.combining(c):
            kinds.add("other")
    if "indic" in kinds:
        return "indic"
    return "latin_accented" if kinds <= {"latin"} else "other"


def skeleton(name_upper):
    """Consonant-skeleton key for matching romanized names against English spellings.

    Apply to BOTH sides after to_latin + uppercase + suffix stripping:
      "PRODUCTS" -> "PRDKTS"  and  "PRODAKTS" -> "PRDKTS"
      "LOGISTICS" -> "LJSTKS" and  "LAJISTIKS" -> "LJSTKS"
    """
    out = []
    for w in name_upper.split():
        w = (w.replace("PH", "F").replace("X", "KS").replace("CK", "K")
              .replace("Q", "K").replace("W", "V").replace("Z", "S"))
        w = re.sub(r"G(?=[EIY])", "J", w)          # soft g
        w = re.sub(r"([^AEIOUY])H", r"\1", w)       # drop aspiration: KH->K, SH->S, TH->T
        w = w.replace("C", "K")
        if w:
            w = ("A" if w[0] in "AEIOUY" else w[0]) + re.sub(r"[AEIOUY]", "", w[1:])
        out.append(re.sub(r"(.)\1+", r"\1", w))
    return " ".join(out)


if __name__ == "__main__":  # quick self-check: python src/translit.py
    for en, loc in [("ADITYA PRODUCTS", "आदित्य प्रोडक्ट्स"), ("HARI LOGISTICS", "ಹರಿ ಲಾಜಿಸ್ಟಿಕ್ಸ್"),
                    ("SUN GLOBAL", "സൺ ഗ്ലോബൽ"), ("MAHARASHTRA", "महाराष्ट्र"), ("LEARNING", "Léarning")]:
        lat = to_latin(loc).upper()
        print(f"{loc:22s} -> {lat:20s} skeleton {skeleton(lat):12s} vs {skeleton(en)}")
