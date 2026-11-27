def build_face_head(name: str):
    """'dfa': the original head.py (kept so existing checkpoints reproduce).
    'aligned': SCRFD detection + 5-point alignment + AdaFace IR-101 (head_aligned.py).
    'buffalo_l': the unmodified upstream MyVLM head, insightface buffalo_l (head_buffalo.py)."""
    if name == 'dfa':
        from concept_heads.face_recognition.head import FaceConceptHead
        return FaceConceptHead()
    if name == 'aligned':
        from concept_heads.face_recognition.head_aligned import AlignedFaceConceptHead
        return AlignedFaceConceptHead()
    if name == 'buffalo_l':
        from concept_heads.face_recognition.head_buffalo import FaceConceptHead
        return FaceConceptHead()
    raise ValueError(f"Unknown face head {name!r}; expected 'dfa', 'aligned' or 'buffalo_l'")
