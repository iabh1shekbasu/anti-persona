"""Anti-Persona: image-level protection against identity binding and recognition in personalized VLMs.

Modules
  config    every hyper-parameter of the final method
  encoder   read_image, ClipEncoder (CLIP ViT-L/14-336, 25 hidden states)
  losses    L_self, L_proto, L_CES and their weighted sum
  bank      leave-one-out token prototype; CES evidence bank (anchors + COCO-100 calibration)
  person    DeepLabV3 person masks -> person patches (anchor selection only)
  face      buffalo_l torch replay, aligned crops, leave-one-out face prototype term
  optim     JPEG in the loop, gradient smoothing, signed descent, bounded rounding
  protect   album-level command line: python -m anti_persona.protect --album <dir> --out <dir>
"""
__version__ = "1.0.0"
