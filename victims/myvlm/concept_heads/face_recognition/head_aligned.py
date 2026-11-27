"""Face concept head with real detection and alignment in front of AdaFace IR-101.

head.py feeds the full-resolution photo straight into the CVLface DFA aligner, which was
trained on 160x160 face-centred crops (its config: input_size 160). On whole scenes it
returns nearly the full frame, sometimes rotated, and its detection score is never
checked, so AdaFace ends up embedding scenery and clothing rather than a face: other
people come out closer than the same person, and no gate threshold separates them.

This head restores the preprocessing AdaFace was trained with, mirroring what upstream
MyVLM gets from insightface buffalo_l:
  1. SCRFD (det_10g) finds every face with its 5 landmarks, score >= DET_THRESHOLD;
  2. each face is warped onto the standard 112x112 ArcFace template (norm_crop);
  3. AdaFace IR-101 WebFace12M embeds the RGB crop in [-1, 1]; embeddings are L2-normed.
Faces are returned largest first, so a photo of one person puts that person at index 0,
which is the face MyVLM uses as the training query. Photos with no face map to None,
exactly as upstream.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from insightface.model_zoo import get_model
from insightface.utils import face_align
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel

from concept_heads.concept_head import ConceptHead

DETECTOR = Path("~/.insightface/models/buffalo_l/det_10g.onnx").expanduser()
RECOGNIZER = Path("~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m").expanduser()
DET_THRESHOLD = 0.5
DET_SIZE = (640, 640)


class AlignedFaceConceptHead(ConceptHead):

    def __init__(self, det_threshold: float = DET_THRESHOLD):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = get_model(str(DETECTOR), providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=-1, input_size=DET_SIZE, det_thresh=det_threshold)
        self.recognizer = self._load_recognizer().eval().to(self.device)
        self.last_detections = {}

    @staticmethod
    def _load_recognizer():
        cwd = os.getcwd()
        try:
            os.chdir(RECOGNIZER)
            sys.path.insert(0, str(RECOGNIZER))
            return AutoModel.from_pretrained(str(RECOGNIZER), trust_remote_code=True)
        finally:
            os.chdir(cwd)
            sys.path.pop(0)

    def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Optional[Tensor]]:
        output = {}
        for path in tqdm(image_paths):
            faces = self.detect(np.array(Image.open(path).convert("RGB")))
            self.last_detections[path] = [(f["score"], f["box"]) for f in faces]
            output[path] = torch.stack([f["embedding"] for f in faces]) if faces else None
        return output

    @torch.no_grad()
    def detect(self, rgb: np.ndarray) -> List[dict]:
        """Every face in the photo, largest first, each with its aligned crop and embedding."""
        boxes, landmarks = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), max_num=0)
        faces = []
        for box, points in zip(boxes, landmarks):
            crop = face_align.norm_crop(rgb, landmark=points, image_size=112)
            x = torch.from_numpy(crop).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
            emb = self.recognizer(x.unsqueeze(0).to(self.device))
            if isinstance(emb, dict):
                emb = emb["embedding"]
            faces.append(dict(score=float(box[4]), box=[float(v) for v in box[:4]],
                              area=float((box[2] - box[0]) * (box[3] - box[1])), crop=crop,
                              embedding=F.normalize(emb.float(), p=2, dim=1)[0].cpu()))
        return sorted(faces, key=lambda f: -f["area"])
