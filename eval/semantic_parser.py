"""Answer parsers used for every number in the paper (pure Python, no model needed).

Yo'LLaVA (question: "<sks> is <token0>...<token15> Can you see <sks> in this photo? Answer with a single word or phrase.")

  semantic_answer(text) -> 'yes' | 'no' | None        parser of every Yo'LLaVA table
      0. normalise: lowercase, drop concept tokens such as <sks>, unify curly apostrophes, collapse whitespace, strip
         leading quotes / punctuation;
      1. leading word: starts with the word "yes" -> yes; starts with the word "no" -> no;
      2. a negation phrase anywhere (NEGATIONS) -> no;
      3. an affirmative phrase anywhere (AFFIRMATIONS) -> yes;
      4. otherwise None (invalid). An invalid answer counts as NOT protected.

  A photo counts as protected (not recognised) when the parsed answer is 'no'.

MyVLM (question: "Can you see if sks is in this photo? Answer with a single word or phrase.", concept identifier "sks")

  myvlm_recognised(text, concept='sks') -> bool
      MyVLM's own probe parser (victims/myvlm/probe_scoring.py: probe_answer_is_recognition), which this function
      calls unchanged: recognised if the answer is affirmative at the start ("yes", "definitely", "correct", ...;
      not "no ..." / "yes, but ...") or it names the identifier outside a refusal context.

  A photo counts as protected (not recognised) when myvlm_recognised(...) is False.

The Yo'LLaVA rules and phrase lists are a verbatim copy of the scoring module that produced the paper tables;
tests/test_semantic_parser.py checks the Yo'LLaVA and MyVLM parsers against answers recorded in those scoring jobs.
"""
import re
import sys
from pathlib import Path

NEGATIONS = ('is not', "isn't", 'not in', 'not visible', 'not present', 'not shown', 'cannot see', "can't see",
             'do not see', "don't see", 'no one', 'not appear', 'absent',
             # phrasings found in an audit of the saved answers
             'not recognize', "n't recognize", 'not spotted', 'not featured', 'does not have')
AFFIRMATIONS = ('is in', 'is visible', 'is present', 'is shown', 'can see', 'appears', 'is here', 'is pictured')

_TOKEN = re.compile(r'<[^<>]*>')
_LEAD = re.compile(r'^[\s"\'`*.,:;!\-]+')
_NEG = re.compile(r'\b(?:' + '|'.join(re.escape(p) for p in NEGATIONS) + r')\b')
_AFF = re.compile(r'\b(?:' + '|'.join(re.escape(p) for p in AFFIRMATIONS) + r')\b')


def normalise(text):
    t = (text or '').lower().replace('’', "'").replace('‘', "'")
    t = _TOKEN.sub(' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return _LEAD.sub('', t)


def semantic_answer(text):
    """'yes' / 'no' / None (invalid) for one raw Yo'LLaVA answer."""
    t = normalise(text)
    if re.match(r'yes\b', t):
        return 'yes'
    if re.match(r'no\b', t):
        return 'no'
    if _NEG.search(t):
        return 'no'
    if _AFF.search(t):
        return 'yes'
    return None


def yollava_protected(text):
    """True when the answer counts as 'not recognised' (the protection numerator)."""
    return semantic_answer(text) == 'no'


_MYVLM_DIR = Path(__file__).resolve().parents[1] / 'victims' / 'myvlm'


def _probe():
    if str(_MYVLM_DIR) not in sys.path:
        sys.path.insert(0, str(_MYVLM_DIR))
    import probe_scoring
    return probe_scoring


def myvlm_recognised(text, concept='sks'):
    """MyVLM probe parser (unchanged): True if the answer recognises the concept."""
    return bool(_probe().probe_answer_is_recognition(concept, text)[0])


def myvlm_protected(text, concept='sks'):
    return not myvlm_recognised(text, concept)
