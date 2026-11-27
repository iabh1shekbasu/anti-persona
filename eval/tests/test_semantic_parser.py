"""CPU test of eval/semantic_parser.py: recorded answers of the paper's scoring jobs and hand-written edge cases.

    python -m pytest eval/tests/test_semantic_parser.py      or      python eval/tests/test_semantic_parser.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eval import semantic_parser as P  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'recorded_answers.json'

# hand-written cases of the original module (expected semantic decision)
EDGE = [
    ('No', 'no'), ('Yes', 'yes'), ('yes.', 'yes'), ('No.', 'no'),
    ('No,  <sks> is not in this photo.', 'no'), ('<sks> is not in this photo.', 'no'),
    ('<sks> is not in the image.', 'no'), ('Yes, <sks> is in this photo.', 'yes'),
    ('<sks> is visible in this photo.', 'yes'), ("<sks> isn't here.", 'no'),
    ('I cannot see <sks> in this image.', 'no'), ('I can see <sks> on the left.', 'yes'),
    ('<sks> appears in the photo.', 'yes'), ('<sks> does not appear in this photo.', 'no'),
    ('There is no one in this photo.', 'no'), ('<sks> is pictured here.', 'yes'),
    ('Nobody knows.', None), ('Not sure.', None), ('', None), (None, None),
    ('"Yes, it is."', 'yes'), ('Note the person on the left.', None), ('Notably, <sks> is in the photo.', 'yes'),
    ('<sks> is absent.', 'no'), ('A man is standing in a park.', None),
    ('I don’t see <sks>.', 'no'), ('<sks> is in this photo.', 'yes'),
    ('I do not recognize  <sks> in this photo.', 'no'), ('I have not spotted  <sks> in this photo.', 'no'),
    ('<sks> is not featured in this photo.', 'no'), ('No, this photo does not have  <sks> .', 'no'),
]


def test_edge_cases():
    bad = [(t, e, P.semantic_answer(t)) for t, e in EDGE if P.semantic_answer(t) != e]
    assert not bad, bad


def test_recorded_yollava():
    cases = json.loads(FIXTURE.read_text())['yollava']
    assert len(cases) >= 30
    for c in cases:
        assert P.semantic_answer(c['answer']) == c['semantic'], c


def test_recorded_myvlm():
    cases = json.loads(FIXTURE.read_text())['myvlm']
    assert len(cases) >= 10
    for c in cases:
        assert P.myvlm_recognised(c['answer'], 'sks') == c['recognised'], c


if __name__ == '__main__':
    test_edge_cases()
    test_recorded_yollava()
    test_recorded_myvlm()
    data = json.loads(FIXTURE.read_text())
    print(f"semantic parser OK: {len(EDGE)} edge cases, {len(data['yollava'])} recorded Yo'LLaVA answers, "
          f"{len(data['myvlm'])} recorded MyVLM answers")
