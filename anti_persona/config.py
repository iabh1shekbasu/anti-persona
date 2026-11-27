"""Hyper-parameters of the final Anti-Persona method.

These constants are the ones that produced every number in the paper. They are mirrored, for reading, in
``configs/final.yaml``; the code below is the source of truth. Pixel quantities are in units of [0, 255].

    L = W_SELF * L_self + W_PROTO * L_proto + W_CES * L_CES + W_FACE * L_face
"""

# Loss weights
W_SELF = 1.0
W_PROTO = 0.5     # token prototype over the summary token and ALL 576 patches (no segmentation mask)
W_CES = 2.0       # Calibrated Evidence Suppression
W_FACE = 0.125    # buffalo_l (w600k_r50) face prototype, fed B,G,R

# Optimisation (signed gradient descent from a zero start)
EPS = 4.0         # l_inf budget: 4/255
STEP = 0.5        # step size: 0.5/255
STEPS = 501

# Robustness
JPEG_CLEAN_P = 0.3        # no JPEG with probability 0.3 (= JPEG in the loop with probability 0.7)
JPEG_Q = (60, 95)         # quality drawn uniformly from these whole numbers (inclusive)
SMOOTH_EVERY = 3          # 3x3 Gaussian (sigma 3) gradient smoothing on steps 3, 6, 9, ...
SMOOTH_SIGMA = 3.0

# CES
TAU = 0.05                # soft-maximum temperature (beta in the paper)
MARGIN = 0.05             # margin m
TOP_SHARE = 0.25          # mean over the strongest 25% of anchors
ANCHORS_PER_PHOTO = 32    # at most 32 person-patch anchors per reference photo, evenly spaced
QUANTILE = 0.9            # anchor threshold = 90th percentile of its response on the COCO-100 calibration set

# Encoder
CLIP_NAME = "openai/clip-vit-large-patch14-336"
IMAGE_SIZE = 336
NUM_ENTRIES = 25          # the tokens before the first block + the output of each of the 24 blocks
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Person masks (DeepLabV3-ResNet101, torchvision COCO-with-VOC-labels weights); only used to choose CES anchors
PERSON_CLASS = 15
PERSON_PROB = 0.5
DILATE_FRACTION = 0.02    # dilation radius = 2% of the long side
TOKEN_FRACTION = 0.25     # a 14x14 patch is "person" when at least 25% of it is person
