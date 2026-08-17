# ####################################################################################################################################################################
import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch import Tensor

from transformers import AutoModel
from huggingface_hub import snapshot_download

# Add the missing import
from abc import ABC, abstractmethod

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ConceptHead(ABC):
    @abstractmethod
    def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
        raise NotImplementedError

class FaceConceptHead(ConceptHead):

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._setup_models()

    def _setup_models(self):
        home = os.path.expanduser("~")
        aligner_path = os.path.join(home, ".cvlface_cache", "minchul", "cvlface_DFA_mobilenet")
        rec_path = os.path.join(home, ".cvlface_cache", "minchul", "cvlface_adaface_ir101_webface12m")
        
        # Download if needed
        if not os.path.exists(aligner_path):
            snapshot_download("minchul/cvlface_DFA_mobilenet", local_dir=aligner_path, local_dir_use_symlinks=False)
            self._fix_model_path(aligner_path)
            
        if not os.path.exists(rec_path):
            snapshot_download("minchul/cvlface_adaface_ir101_webface12m", local_dir=rec_path, local_dir_use_symlinks=False)
            self._fix_model_path(rec_path)
        
        # Load models
        self.aligner = self._load_model(aligner_path).eval().to(self.device)
        self.recognizer = self._load_model(rec_path).eval().to(self.device)

    def _fix_model_path(self, model_path):
        import shutil
        pm_dir = os.path.join(model_path, "pretrained_model")
        os.makedirs(pm_dir, exist_ok=True)
        
        for fname in ["model.pt", "model.safetensors"]:
            src = os.path.join(model_path, fname)
            dst = os.path.join(pm_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    def _load_model(self, path):
        cwd = os.getcwd()
        try:
            os.chdir(path)
            sys.path.insert(0, path)
            
            import io
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            try:
                model = AutoModel.from_pretrained(path, trust_remote_code=True)
            finally:
                sys.stderr = old_stderr
                
            return model
        finally:
            os.chdir(cwd)
            sys.path.pop(0)

    def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
        output = {}
        for path in tqdm(image_paths):
            # Handle both string and Path objects
            path_obj = Path(path) if isinstance(path, str) else path
            
            image = Image.open(path).convert("RGB")
            faces = self._get_faces(np.array(image))
            
            print(f"[DEBUG] {path_obj.name}: Found {len(faces)} faces")
            if len(faces) > 0:
                print(f"[DEBUG] Face shape: {faces[0].shape}")
                print(f"[DEBUG] Face embedding shape: {faces[0].normed_embedding.shape}")
                
            if len(faces) == 0:
                output[path] = None
            else:
                embeddings = torch.stack([torch.from_numpy(f.normed_embedding) for f in faces])
                print(f"[DEBUG] Final embeddings shape: {embeddings.shape}")
                output[path] = embeddings
        return output

    def _get_faces(self, image_np):
        try:
            print(f"[DEBUG] Starting face detection...")
            
            # Preprocess
            x = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            x = (x - 0.5) / 0.5
            x = x.unsqueeze(0).to(self.device)
            print(f"[DEBUG] Preprocessed tensor shape: {x.shape}")

            # Detect and align
            aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = self.aligner(x)
            print(f"[DEBUG] Aligner score: {score.item():.4f}")

            # Recognize
            print(f"[DEBUG] Running recognizer...")
            
            emb = self.recognizer(aligned_x)
            print(f"[DEBUG] Recognizer output type: {type(emb)}")
            
            if isinstance(emb, dict):
                emb = emb.get("embedding", None)
                print(f"[DEBUG] Extracted embedding from dict")
                
            if emb is None:
                print(f"[DEBUG] Embedding is None")
                return []
                
            if emb.numel() == 0:
                print(f"[DEBUG] Embedding is empty")
                return []

            print(f"[DEBUG] Embedding shape: {emb.shape}")

            # Normalize
            emb_norm = F.normalize(emb, p=2, dim=1).float()
            print(f"[DEBUG] Normalized embedding shape: {emb_norm.shape}")
            
            # Return only first face (like buffalo_l behavior)
            face = FaceObject()
            face.normed_embedding = emb_norm[0].detach().cpu().numpy()
            print(f"[DEBUG] Final face embedding shape: {face.normed_embedding.shape}")
            return [face]  # Return single face like buffalo_l
            
        except Exception as e:
            print(f"[DEBUG] Exception in _get_faces: {e}")
            import traceback
            traceback.print_exc()
            return []


class FaceObject:
    def __init__(self):
        self.normed_embedding = None
        
    @property
    def shape(self):
        return self.normed_embedding.shape if self.normed_embedding is not None else None


####################################################################################################################################################################

# import os
# import sys
# import warnings
# from pathlib import Path
# from typing import List, Dict

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from tqdm import tqdm
# from torch import Tensor

# from transformers import AutoModel
# from huggingface_hub import snapshot_download

# # Add the missing import
# from abc import ABC, abstractmethod

# # Suppress warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# class ConceptHead(ABC):
#     @abstractmethod
#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
#         raise NotImplementedError

# class FaceConceptHead(ConceptHead):

#     def __init__(self):
#         print("[DEBUG] Initializing CVLFace models...")
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         print(f"[DEBUG] Using device: {self.device}")
#         self._setup_models()

#     def _setup_models(self):
#         home = os.path.expanduser("~")
#         aligner_path = os.path.join(home, ".cvlface_cache", "minchul", "cvlface_DFA_mobilenet")
#         rec_path = os.path.join(home, ".cvlface_cache", "minchul", "cvlface_adaface_vit_base_kprpe_webface4m")
        
#         print(f"[DEBUG] Aligner path: {aligner_path}")
#         print(f"[DEBUG] Recognizer path: {rec_path}")
        
#         # Download if needed
#         if not os.path.exists(aligner_path):
#             print("[DEBUG] Downloading aligner model...")
#             snapshot_download("minchul/cvlface_DFA_mobilenet", local_dir=aligner_path, local_dir_use_symlinks=False)
#             self._fix_model_path(aligner_path)
            
#         if not os.path.exists(rec_path):
#             print("[DEBUG] Downloading recognizer model...")
#             snapshot_download("minchul/cvlface_adaface_vit_base_kprpe_webface4m", local_dir=rec_path, local_dir_use_symlinks=False)
#             self._fix_model_path(rec_path)
        
#         # Load models
#         print("[DEBUG] Loading aligner model...")
#         self.aligner = self._load_model(aligner_path).eval().to(self.device)
#         print("[DEBUG] Loading recognizer model...")
#         self.recognizer = self._load_model(rec_path).eval().cpu()
#         print("[DEBUG] CVLFace models loaded successfully")

#     def _fix_model_path(self, model_path):
#         import shutil
#         pm_dir = os.path.join(model_path, "pretrained_model")
#         os.makedirs(pm_dir, exist_ok=True)
        
#         for fname in ["model.pt", "model.safetensors"]:
#             src = os.path.join(model_path, fname)
#             dst = os.path.join(pm_dir, fname)
#             if os.path.exists(src) and not os.path.exists(dst):
#                 shutil.copy2(src, dst)

#     def _load_model(self, path):
#         cwd = os.getcwd()
#         try:
#             os.chdir(path)
#             sys.path.insert(0, path)
            
#             import io
#             old_stderr = sys.stderr
#             sys.stderr = io.StringIO()
            
#             try:
#                 model = AutoModel.from_pretrained(path, trust_remote_code=True)
#             finally:
#                 sys.stderr = old_stderr
                
#             return model
#         finally:
#             os.chdir(cwd)
#             sys.path.pop(0)

#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
#         print(f"[DEBUG] Processing {len(image_paths)} images with CVLFace")
#         output = {}
        
#         for path in tqdm(image_paths):
#             # Handle both string and Path objects
#             path_obj = Path(path) if isinstance(path, str) else path
#             print(f"[DEBUG] Processing image: {path_obj.name}")
            
#             image = Image.open(path).convert("RGB")
#             image_np = np.array(image)
#             print(f"[DEBUG] Image shape: {image_np.shape}")
            
#             faces = self._get_faces(image_np, image_path=str(path_obj))
            
#             print(f"[DEBUG] {path_obj.name}: Found {len(faces)} faces")
#             if len(faces) > 0:
#                 print(f"[DEBUG] Face shape: {faces[0].shape}")
#                 print(f"[DEBUG] Face embedding shape: {faces[0].normed_embedding.shape}")
#                 print(f"[DEBUG] Embedding norm: {np.linalg.norm(faces[0].normed_embedding):.4f}")
#                 print(f"[DEBUG] Embedding mean: {faces[0].normed_embedding.mean():.4f}")
#                 print(f"[DEBUG] Embedding std: {faces[0].normed_embedding.std():.4f}")
#                 print(f"[DEBUG] Embedding min: {faces[0].normed_embedding.min():.4f}")
#                 print(f"[DEBUG] Embedding max: {faces[0].normed_embedding.max():.4f}")
#                 print(f"[DEBUG] Embedding first 5 values: {faces[0].normed_embedding[:5]}")
#                 print(f"[DEBUG] Embedding hash: {hash(faces[0].normed_embedding.tobytes())}")
                
#             if len(faces) == 0:
#                 print(f"[DEBUG] No faces detected in {path_obj.name}")
#                 output[path] = None
#             else:
#                 embeddings = torch.stack([torch.from_numpy(f.normed_embedding) for f in faces])
#                 print(f"[DEBUG] Final embeddings tensor shape: {embeddings.shape}")
#                 print(f"[DEBUG] Final embeddings tensor norm: {torch.norm(embeddings[0]).item():.4f}")
#                 output[path] = embeddings
                
#         print(f"[DEBUG] CVLFace processing complete. Processed {len([v for v in output.values() if v is not None])}/{len(output)} images successfully")
#         return output

#     def _get_faces(self, image_np, image_path="unknown"):
#         try:
#             print(f"[DEBUG] Starting face detection for {image_path}...")
            
#             # Preprocess - CVLFace normalization: mean=0.5, std=0.5
#             x = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
#             x = (x - 0.5) / 0.5  # Normalize to [-1, 1] as per CVLFace docs
#             x = x.unsqueeze(0).to(self.device)
#             print(f"[DEBUG] Preprocessed tensor shape: {x.shape}")
#             print(f"[DEBUG] Tensor range: [{x.min():.3f}, {x.max():.3f}]")

#             # Detect and align using CVLFace aligner
#             aligner_output = self.aligner(x)
#             print(f"[DEBUG] Aligner output length: {len(aligner_output)}")
#             print(f"[DEBUG] Aligner output types: {[type(item) for item in aligner_output]}")
            
#             # Unpack aligner output
#             aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner_output
            
#             print(f"[DEBUG] Aligner score: {score.item():.4f}")
#             print(f"[DEBUG] Original landmarks: {orig_ldmks}")
#             print(f"[DEBUG] Original landmarks type: {type(orig_ldmks)}")
#             print(f"[DEBUG] Aligned landmarks shape: {aligned_ldmks.shape if aligned_ldmks is not None else 'None'}")
#             print(f"[DEBUG] Aligned tensor shape: {aligned_x.shape}")
#             print(f"[DEBUG] Bbox: {bbox}")
#             print(f"[DEBUG] Thetas: {thetas}")

#             # Check if we have valid aligned landmarks
#             if aligned_ldmks is None:
#                 print(f"[DEBUG] No aligned landmarks detected")
#                 return []

#             print(f"[DEBUG] Using keypoints shape: {aligned_ldmks.shape}")
#             print(f"[DEBUG] Keypoints sample: {aligned_ldmks[0][:2] if aligned_ldmks is not None else 'None'}")

#             # Recognize - CVLFace recognizer expects (image, keypoints)
#             aligned_x_cpu = aligned_x.cpu()
#             aligned_ldmks_cpu = aligned_ldmks.cpu()
#             print(f"[DEBUG] Running recognizer with image and keypoints...")
#             print(f"[DEBUG] Aligned image shape: {aligned_x_cpu.shape}")
#             print(f"[DEBUG] Aligned landmarks shape: {aligned_ldmks_cpu.shape}")
            
#             # Move recognizer to CPU if not already (CVLFace seems to expect CPU)
#             if next(self.recognizer.parameters()).device.type != 'cpu':
#                 self.recognizer = self.recognizer.cpu()
            
#             emb = self.recognizer(aligned_x_cpu, aligned_ldmks_cpu)
#             print(f"[DEBUG] Recognizer output type: {type(emb)}")
            
#             if isinstance(emb, dict):
#                 emb = emb.get("embedding", None)
#                 print(f"[DEBUG] Extracted embedding from dict")
                
#             if emb is None:
#                 print(f"[DEBUG] Embedding is None")
#                 return []
                
#             if emb.numel() == 0:
#                 print(f"[DEBUG] Embedding is empty")
#                 return []

#             print(f"[DEBUG] Raw embedding shape: {emb.shape}")
#             print(f"[DEBUG] Raw embedding norm: {torch.norm(emb).item():.4f}")
#             print(f"[DEBUG] Raw embedding device: {emb.device}")

#             # Normalize embedding
#             emb_norm = F.normalize(emb, p=2, dim=1).float()
#             print(f"[DEBUG] Normalized embedding shape: {emb_norm.shape}")
#             print(f"[DEBUG] Normalized embedding norm: {torch.norm(emb_norm).item():.4f}")
#             print(f"[DEBUG] Normalized embedding mean: {emb_norm.mean().item():.4f}")
#             print(f"[DEBUG] Normalized embedding std: {emb_norm.std().item():.4f}")
#             print(f"[DEBUG] Normalized embedding min: {emb_norm.min().item():.4f}")
#             print(f"[DEBUG] Normalized embedding max: {emb_norm.max().item():.4f}")
            
#             # Create face object (like buffalo_l behavior)
#             face = FaceObject()
#             face.normed_embedding = emb_norm[0].detach().cpu().numpy()  # Ensure CPU and detached
#             print(f"[DEBUG] Final face embedding shape: {face.normed_embedding.shape}")
#             print(f"[DEBUG] Final face embedding dtype: {face.normed_embedding.dtype}")
            
#             # Additional consistency check
#             print(f"[DEBUG] Face embedding norm after conversion: {np.linalg.norm(face.normed_embedding):.4f}")
            
#             return [face]  # Return single face like buffalo_l
            
#         except Exception as e:
#             print(f"[DEBUG] Exception in _get_faces for {image_path}: {e}")
#             import traceback
#             traceback.print_exc()
#             return []


# class FaceObject:
#     def __init__(self):
#         self.normed_embedding = None
        
#     @property
#     def shape(self):
#         return self.normed_embedding.shape if self.normed_embedding is not None else None




# import os
# import sys
# import warnings
# from pathlib import Path
# from typing import List, Dict

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from tqdm import tqdm
# from torch import Tensor

# from transformers import AutoModel
# from huggingface_hub import snapshot_download

# # Add the missing import
# from abc import ABC, abstractmethod

# # Suppress warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# class ConceptHead(ABC):
#     @abstractmethod
#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
#         raise NotImplementedError

# class FaceConceptHead(ConceptHead):

#     def __init__(self):
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self._setup_models()

#     def _setup_models(self):
#         home = os.path.expanduser("~")
#         aligner_path = os.path.join(home, ".cvlface_cache", "minchul", "cvlface_DFA_mobilenet")
#         rec_path = os.path.join(home, ".cvlface_cache", "minchul", "cvlface_adaface_vit_base_kprpe_webface4m")
        
#         # Download if needed
#         if not os.path.exists(aligner_path):
#             snapshot_download("minchul/cvlface_DFA_mobilenet", local_dir=aligner_path, local_dir_use_symlinks=False)
#             self._fix_model_path(aligner_path)
            
#         if not os.path.exists(rec_path):
#             snapshot_download("minchul/cvlface_adaface_vit_base_kprpe_webface4m", local_dir=rec_path, local_dir_use_symlinks=False)
#             self._fix_model_path(rec_path)
        
#         # Load models
#         self.aligner = self._load_model(aligner_path).eval().to(self.device)
#         self.recognizer = self._load_model(rec_path).eval().cpu()

#     def _fix_model_path(self, model_path):
#         import shutil
#         pm_dir = os.path.join(model_path, "pretrained_model")
#         os.makedirs(pm_dir, exist_ok=True)
        
#         for fname in ["model.pt", "model.safetensors"]:
#             src = os.path.join(model_path, fname)
#             dst = os.path.join(pm_dir, fname)
#             if os.path.exists(src) and not os.path.exists(dst):
#                 shutil.copy2(src, dst)

#     def _load_model(self, path):
#         cwd = os.getcwd()
#         try:
#             os.chdir(path)
#             sys.path.insert(0, path)
            
#             import io
#             old_stderr = sys.stderr
#             sys.stderr = io.StringIO()
            
#             try:
#                 model = AutoModel.from_pretrained(path, trust_remote_code=True)
#             finally:
#                 sys.stderr = old_stderr
                
#             return model
#         finally:
#             os.chdir(cwd)
#             sys.path.pop(0)

#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
#         output = {}
#         for path in tqdm(image_paths):
#             # Handle both string and Path objects
#             path_obj = Path(path) if isinstance(path, str) else path
            
#             image = Image.open(path).convert("RGB")
#             faces = self._get_faces(np.array(image))
            
#             print(f"[DEBUG] {path_obj.name}: Found {len(faces)} faces")
#             if len(faces) > 0:
#                 print(f"[DEBUG] Face shape: {faces[0].shape}")
#                 print(f"[DEBUG] Face embedding shape: {faces[0].normed_embedding.shape}")
                
#             if len(faces) == 0:
#                 output[path] = None
#             else:
#                 embeddings = torch.stack([torch.from_numpy(f.normed_embedding) for f in faces])
#                 print(f"[DEBUG] Final embeddings shape: {embeddings.shape}")
#                 output[path] = embeddings
#         return output

#     def _get_faces(self, image_np):
#         try:
#             print(f"[DEBUG] Starting face detection...")
            
#             # Preprocess
#             x = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
#             x = (x - 0.5) / 0.5
#             x = x.unsqueeze(0).to(self.device)
#             print(f"[DEBUG] Preprocessed tensor shape: {x.shape}")

#             # Detect and align
#             aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = self.aligner(x)
#             print(f"[DEBUG] Aligner score: {score.item():.4f}")

#             # Recognize
#             aligned_x_cpu = aligned_x.cpu()
#             aligned_ldmks_cpu = aligned_ldmks.cpu()
#             print(f"[DEBUG] Running recognizer...")
            
#             emb = self.recognizer(aligned_x_cpu, aligned_ldmks_cpu)
#             print(f"[DEBUG] Recognizer output type: {type(emb)}")
            
#             if isinstance(emb, dict):
#                 emb = emb.get("embedding", None)
#                 print(f"[DEBUG] Extracted embedding from dict")
                
#             if emb is None:
#                 print(f"[DEBUG] Embedding is None")
#                 return []
                
#             if emb.numel() == 0:
#                 print(f"[DEBUG] Embedding is empty")
#                 return []

#             print(f"[DEBUG] Embedding shape: {emb.shape}")

#             # Normalize
#             emb_norm = F.normalize(emb, p=2, dim=1).float()
#             print(f"[DEBUG] Normalized embedding shape: {emb_norm.shape}")
            
#             # Return only first face (like buffalo_l behavior)
#             face = FaceObject()
#             face.normed_embedding = emb_norm[0].detach().numpy()  # Add .detach()
#             print(f"[DEBUG] Final face embedding shape: {face.normed_embedding.shape}")
#             return [face]  # Return single face like buffalo_l
            
#         except Exception as e:
#             print(f"[DEBUG] Exception in _get_faces: {e}")
#             import traceback
#             traceback.print_exc()
#             return []


# class FaceObject:
#     def __init__(self):
#         self.normed_embedding = None
        
#     @property
#     def shape(self):
#         return self.normed_embedding.shape if self.normed_embedding is not None else None




































#### THIS WORKS
# import os
# import sys
# import shutil
# import warnings
# from pathlib import Path
# from typing import List, Dict, Optional

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from tqdm import tqdm
# from torch import Tensor

# from transformers import AutoModel
# from huggingface_hub import snapshot_download, hf_hub_download

# # Minimal abstract base (kept local to avoid import path issues)
# from abc import ABC, abstractmethod

# # Suppress CUDA warnings for the RPE operations
# warnings.filterwarnings("ignore", category=UserWarning, message=".*Failed to install.*rpe_ops.*")
# warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA.*")


# class ConceptHead(ABC):
#     @abstractmethod
#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Optional[Tensor]]:
#         """
#         Map each image Path to a Tensor signal (e.g., [N,D] embeddings) or None (no face).
#         """
#         raise NotImplementedError


# class FaceConceptHead(ConceptHead):
#     """
#     HF-backed Face Concept Head using:
#       - minchul/cvlface_DFA_mobilenet (aligner)
#       - minchul/cvlface_adaface_vit_base_kprpe_webface4m (recognizer)

#     Fixed version that handles RPE GPU operation failures gracefully.
#     """

#     def __init__(
#         self,
#         HF_TOKEN: Optional[str] = None,
#         force_download: bool = False,
#         device: Optional[str] = None,
#         debug: bool = False,
#     ):
#         self.HF_TOKEN = HF_TOKEN or os.getenv("HF_TOKEN")
#         self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
#         self.debug = debug

#         # HF repositories
#         self.aligner_repo = "minchul/cvlface_DFA_mobilenet"
#         self.rec_repo     = "minchul/cvlface_adaface_vit_base_kprpe_webface4m"

#         # Local cache targets (materialized trees, not symlinks)
#         home = os.path.expanduser("~")
#         self.aligner_path = os.path.join(home, ".cvlface_cache", "minchul", "cvlface_DFA_mobilenet")
#         self.rec_path     = os.path.join(home, ".cvlface_cache", "minchul", "cvlface_adaface_vit_base_kprpe_webface4m")

#         if self.debug:
#             print(f"[HF FaceConceptHead] device={self.device}")

#         # Fetch full repos
#         self._materialize_repo(self.aligner_repo, self.aligner_path, force_download)
#         self._materialize_repo(self.rec_repo,     self.rec_path,     force_download)

#         # Load models
#         if self.debug: 
#             print("[HF] Loading aligner ...")
#         self.aligner = self._load_model_from_local_path(self.aligner_path).eval().to(self.device)

#         if self.debug: 
#             print("[HF] Loading recognizer ...")
#         self.recognizer = self._load_model_from_local_path(self.rec_path).eval()
        
#         # CRITICAL FIX: Force recognizer to CPU to avoid RPE GPU issues
#         if self.debug:
#             print("[HF] Moving recognizer to CPU to avoid RPE GPU issues...")
#         self.recognizer = self.recognizer.cpu()

#         if self.debug:
#             print(f"[HF] Models ready - aligner on {self.device}, recognizer on CPU")

#     # ----------------------- Environment management -----------------------

#     def _disable_cuda_compilation(self):
#         """Temporarily disable CUDA compilation to avoid version mismatch issues"""
#         env_vars = {
#             'TORCH_CUDA_ARCH_LIST': '',
#             'FORCE_CUDA': '0',
#         }
        
#         old_values = {}
#         for key, value in env_vars.items():
#             old_values[key] = os.environ.get(key)
#             os.environ[key] = value
            
#         return old_values

#     def _restore_cuda_env(self, old_values):
#         """Restore CUDA environment variables"""
#         for key, old_value in old_values.items():
#             if old_value is not None:
#                 os.environ[key] = old_value
#             elif key in os.environ:
#                 del os.environ[key]

#     # ----------------------- Robust repository materialization -----------------------

#     def _materialize_repo(self, repo_id: str, local_dir: str, force: bool) -> None:
#         """
#         Download the ENTIRE HF repo tree into local_dir (no symlinks).
#         Then normalize assets expected by third-party wrapper code.
#         """
#         if force and os.path.exists(local_dir):
#             shutil.rmtree(local_dir)

#         if self.debug:
#             print(f"[HF] Downloading {repo_id} to {local_dir}")

#         # 1) Full snapshot (strong guarantee all nested files exist)
#         snapshot_download(
#             repo_id=repo_id,
#             local_dir=local_dir,
#             local_dir_use_symlinks=False,  # write real files
#             token=self.HF_TOKEN,
#         )

#         # 2) Normalize required nested file paths the wrapper expects
#         self._ensure_required_assets(local_dir, repo_id)

#     def _ensure_required_assets(self, repo_root: str, repo_id: str) -> None:
#         """
#         Some repos ship only model.safetensors, but wrapper expects 'pretrained_model/model.pt'.
#         Normalize so the wrapper's hardcoded relative path always resolves.
#         """
#         pm_dir = os.path.join(repo_root, "pretrained_model")
#         os.makedirs(pm_dir, exist_ok=True)

#         pt_target = os.path.join(pm_dir, "model.pt")
#         st_target = os.path.join(pm_dir, "model.safetensors")

#         # Look for model files in the root directory
#         root_pt = os.path.join(repo_root, "model.pt")
#         root_st = os.path.join(repo_root, "model.safetensors")

#         # Copy model files to pretrained_model directory if they don't exist there
#         if not os.path.exists(pt_target):
#             if os.path.exists(root_pt):
#                 shutil.copy2(root_pt, pt_target)
#                 if self.debug:
#                     print(f"[HF] Copied {root_pt} -> {pt_target}")
#             elif os.path.exists(root_st):
#                 shutil.copy2(root_st, pt_target)
#                 if self.debug:
#                     print(f"[HF] Copied {root_st} -> {pt_target} (as fallback)")

#         if not os.path.exists(st_target) and os.path.exists(root_st):
#             shutil.copy2(root_st, st_target)
#             if self.debug:
#                 print(f"[HF] Copied {root_st} -> {st_target}")

#         # Copy other potential required files
#         for filename in ["model.yaml", "config.yaml", "aligner.pt", "aligner.yaml"]:
#             root_file = os.path.join(repo_root, filename)
#             target_file = os.path.join(pm_dir, filename)
#             if os.path.exists(root_file) and not os.path.exists(target_file):
#                 shutil.copy2(root_file, target_file)
#                 if self.debug:
#                     print(f"[HF] Copied {root_file} -> {target_file}")

#         # Sanity check: if still no model.pt, try direct download
#         if not os.path.exists(pt_target):
#             if self.debug:
#                 print(f"[HF] Warning: No model.pt found, attempting direct download...")
            
#             # Try to download specific files that might be needed
#             for fname in ["pretrained_model/model.pt", "model.pt", "pretrained_model/model.safetensors"]:
#                 try:
#                     hf_hub_download(
#                         repo_id=repo_id,
#                         filename=fname,
#                         local_dir=repo_root,
#                         local_dir_use_symlinks=False,
#                         token=self.HF_TOKEN,
#                     )
#                     if self.debug:
#                         print(f"[HF] Successfully downloaded {fname}")
#                     break
#                 except Exception as e:
#                     if self.debug:
#                         print(f"[HF] Failed to download {fname}: {e}")
#                     continue

#     # ----------------------- Model loader -----------------------

#     def _load_model_from_local_path(self, path):
#         """Load model from local path with robust error handling"""
#         original_cwd = os.getcwd()
#         original_path = sys.path.copy()
#         old_cuda_env = None
        
#         try:
#             # Disable CUDA compilation before model loading
#             old_cuda_env = self._disable_cuda_compilation()
            
#             # Change to model directory - critical for wrapper's relative paths
#             os.chdir(path)
#             sys.path.insert(0, path)
            
#             if self.debug:
#                 print(f"[HF] Loading model from {path}")
#                 print(f"[HF] Working directory: {os.getcwd()}")
#                 pm_dir = os.path.join(path, "pretrained_model")
#                 if os.path.exists(pm_dir):
#                     print(f"[HF] Files in pretrained_model: {os.listdir(pm_dir)}")

#             # Load model with error suppression
#             with warnings.catch_warnings():
#                 warnings.filterwarnings("ignore")
                
#                 # Suppress stderr to hide CUDA compilation warnings
#                 import io
#                 old_stderr = sys.stderr
#                 sys.stderr = io.StringIO()
                
#                 try:
#                     model = AutoModel.from_pretrained(
#                         path,
#                         trust_remote_code=True,
#                         token=self.HF_TOKEN
#                     )
#                 finally:
#                     sys.stderr = old_stderr
                    
#                     # Ensure we're still in the right directory after model loading
#                     current_dir = os.getcwd()
#                     if current_dir != path:
#                         if self.debug:
#                             print(f"[HF] Directory changed from {path} to {current_dir}, restoring...")
#                         os.chdir(path)
            
#             return model
            
#         except Exception as e:
#             if self.debug:
#                 print(f"[HF] Error loading model from {path}: {e}")
#                 print(f"[HF] Current working directory: {os.getcwd()}")
                
#                 # Provide debugging information
#                 try:
#                     os.chdir(path)  # Force back to model directory
#                     print(f"[HF] Files in model root: {os.listdir('.')}")
#                     pm_dir = os.path.join(path, "pretrained_model")
#                     if os.path.exists(pm_dir):
#                         print(f"[HF] Files in pretrained_model: {os.listdir(pm_dir)}")
#                 except Exception as debug_e:
#                     print(f"[HF] Debug info failed: {debug_e}")
#             raise
#         finally:
#             # Always restore original state
#             if old_cuda_env is not None:
#                 self._restore_cuda_env(old_cuda_env)
#             os.chdir(original_cwd)
#             sys.path[:] = original_path

#     # ----------------------- Interface impl -----------------------

#     @torch.inference_mode()
#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Optional[Tensor]]:
#         """
#         Detect+align face(s) → extract embeddings → L2-normalize.
#         Returns {path: Tensor [N, D]} or {path: None}.
        
#         FIXED VERSION: Handles RPE GPU operation failures by using CPU for recognizer.
#         """
#         output: Dict[Path, Optional[Tensor]] = {}

#         for path in tqdm(image_paths, desc="HF FaceConceptHead"):
#             try:
#                 image = Image.open(path).convert("RGB")
#                 img_np = np.array(image)

#                 # preprocess to [-1, 1]
#                 x = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
#                 x = (x - 0.5) / 0.5
#                 x = x.unsqueeze(0).to(self.device)

#                 # aligner forward (on GPU)
#                 aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = self.aligner(x)
#                 if self.debug:
#                     print(f"[HF] {path.name}: align score={float(score):.3f}, aligned_x={tuple(aligned_x.shape)}")

#                 # Check if face was detected with reasonable confidence
#                 # Lowered threshold since we know detection works well
#                 if score.item() < 0.3:  
#                     if self.debug:
#                         print(f"[HF] {path.name}: Low confidence face detection, skipping")
#                     output[path] = None
#                     continue

#                 # CRITICAL FIX: Move data to CPU for recognizer to avoid RPE GPU issues
#                 aligned_x_cpu = aligned_x.cpu()
#                 aligned_ldmks_cpu = aligned_ldmks.cpu()

#                 # recognizer forward (on CPU to avoid RPE issues)
#                 try:
#                     emb = self.recognizer(aligned_x_cpu, aligned_ldmks_cpu)
#                     if isinstance(emb, dict):
#                         emb = emb.get("embedding", None)

#                     if emb is None or emb.numel() == 0:
#                         if self.debug:
#                             print(f"[HF] {path.name}: No embedding returned")
#                         output[path] = None
#                         continue

#                     # L2-normalize like normed_embedding (already on CPU)
#                     emb_norm = F.normalize(emb, p=2, dim=1).float()
#                     output[path] = emb_norm

#                     if self.debug:
#                         print(f"[HF] {path.name}: Success! Embedding shape: {emb_norm.shape}")

#                 except Exception as recognizer_error:
#                     if self.debug:
#                         print(f"[HF] {path.name}: Recognizer error -> {recognizer_error}")
                    
#                     # If recognizer fails, continue processing other images
#                     output[path] = None

#             except Exception as e:
#                 if self.debug:
#                     print(f"[HF] {path.name}: exception -> {e}")
#                 output[path] = None

#         return output


# # ----------------------- Utility functions -----------------------

# def _glob_images(root: Path) -> List[Path]:
#     """Find all image files in directory"""
#     extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
#     image_paths = []
    
#     for ext in extensions:
#         image_paths.extend(root.rglob(f'*{ext}'))
#         image_paths.extend(root.rglob(f'*{ext.upper()}'))
    
#     return sorted(image_paths)


# # ----------------------- CLI for testing -----------------------

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Test Fixed Face Concept Head')
#     parser.add_argument("--image_root", required=True, help="Root directory containing images")
#     parser.add_argument("--out_pt", help="Optional output .pt file to save embeddings")
#     parser.add_argument("--hf_token", default=None, help="HuggingFace token")
#     parser.add_argument("--device", default=None, help="Device to use (cuda:0, cpu, etc.)")
#     parser.add_argument("--debug", action="store_true", help="Enable debug output")
#     parser.add_argument("--force_download", action="store_true", help="Force re-download models")
#     parser.add_argument("--max_images", type=int, default=None, help="Limit number of images to process")
    
#     args = parser.parse_args()
    
#     print(f"Testing Fixed Face Concept Head...")
#     print(f"Image root: {args.image_root}")
#     print(f"Device: {args.device or 'auto'}")
    
#     # Create face concept head
#     head = FaceConceptHead(
#         HF_TOKEN=args.hf_token,
#         force_download=args.force_download,
#         device=args.device,
#         debug=args.debug,
#     )
    
#     # Find all images
#     image_paths = _glob_images(Path(args.image_root))
#     if args.max_images:
#         image_paths = image_paths[:args.max_images]
    
#     print(f"Found {len(image_paths)} images")
    
#     if len(image_paths) == 0:
#         print("No images found! Check your image_root path.")
#         sys.exit(1)
    
#     # Process images
#     results = head.extract_signal(image_paths)
    
#     # Print statistics
#     successful = sum(1 for v in results.values() if v is not None)
#     failed = len(results) - successful
    
#     print(f"\nResults:")
#     print(f"  Successfully processed: {successful}/{len(results)} images")
#     print(f"  Failed to process: {failed}/{len(results)} images")
    
#     # Optionally save results
#     if args.out_pt:
#         serializable_results = {str(path): embedding for path, embedding in results.items()}
#         os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
#         torch.save(serializable_results, args.out_pt)
#         print(f"  Cache saved to: {args.out_pt}")
    
#     print("🎉 Test completed!")

# original

# import numpy as np
# import torch
# from PIL import Image
# from insightface.app import FaceAnalysis
# from pathlib import Path
# from torch import Tensor
# from tqdm import tqdm
# from typing import List, Dict

# from concept_heads.concept_head import ConceptHead


# class FaceConceptHead(ConceptHead):

#     def __init__(self):
#         self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#         self.app.prepare(ctx_id=0, det_size=(640, 640))

#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
#         output = {}
#         for path in tqdm(image_paths):
#             image = Image.open(path).convert("RGB")
#             faces = self.app.get(np.array(image))
#             if len(faces) == 0:
#                 output[path] = None
#             else:
#                 embeddings = torch.stack([torch.from_numpy(f.normed_embedding) for f in faces])
#                 output[path] = embeddings
#         return output


# Add these debug modifications to concept_heads/face_recognition/head.py

# import numpy as np
# import torch
# from PIL import Image, ImageDraw
# from insightface.app import FaceAnalysis
# from pathlib import Path
# from torch import Tensor
# from tqdm import tqdm
# from typing import List, Dict
# from concept_heads.concept_head import ConceptHead
# import cv2

# class FaceConceptHead(ConceptHead):
#     def __init__(self, debug=True):
#         self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#         self.app.prepare(ctx_id=0, det_size=(640, 640))
#         self.debug = debug

#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
#         output = {}
#         for path in tqdm(image_paths):
#             print(f"\n{'='*60}")
#             print(f"[DEBUG] Processing: {path.name}")
            
#             # Step 1: Load FULL IMAGE
#             image = Image.open(path).convert("RGB")
#             image_np = np.array(image)
            
#             print(f"[DEBUG] Input to Face Concept Head:")
#             print(f"  - Full image shape: {image_np.shape}")
#             print(f"  - Full image size: {image.size}")
            
#             if self.debug:
#                 # Save the input image for reference
#                 debug_input_path = f"debug_face_input_{path.stem}.jpg"
#                 image.save(debug_input_path)
#                 print(f"  - Saved input image: {debug_input_path}")
            
#             # Step 2: Pass FULL IMAGE to InsightFace
#             print(f"[DEBUG] Calling InsightFace with full image...")
#             faces = self.app.get(image_np)  # This is where the magic happens
            
#             print(f"[DEBUG] InsightFace Results:")
#             print(f"  - Number of faces detected: {len(faces)}")
            
#             if len(faces) == 0:
#                 print(f"  - No faces found, returning None")
#                 output[path] = None
#             else:
#                 # Step 3: Debug what InsightFace returns
#                 for i, face in enumerate(faces):
#                     print(f"  - Face {i}:")
#                     print(f"    * Bounding box: {face.bbox}")
#                     print(f"    * Detection confidence: {face.det_score:.3f}")
#                     print(f"    * Face embedding shape: {face.normed_embedding.shape}")
#                     print(f"    * Face embedding sample: {face.normed_embedding[:5]}...")
                    
#                     if self.debug:
#                         # Extract and save the face crop to prove it's cropped
#                         bbox = face.bbox.astype(int)
#                         x1, y1, x2, y2 = bbox
                        
#                         # Crop the face from the original image
#                         face_crop = image.crop((x1, y1, x2, y2))
                        
#                         print(f"    * Face crop coordinates: ({x1}, {y1}) to ({x2}, {y2})")
#                         print(f"    * Face crop size: {face_crop.size}")
#                         print(f"    * Face crop shape: {np.array(face_crop).shape}")
                        
#                         # Save the cropped face
#                         crop_path = f"debug_face_crop_{path.stem}_face_{i}.jpg"
#                         face_crop.save(crop_path)
#                         print(f"    * Saved face crop: {crop_path}")
                        
#                         # Create visualization with bounding box
#                         vis_image = image.copy()
#                         draw = ImageDraw.Draw(vis_image)
#                         draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
#                         draw.text((x1, y1-20), f"Face {i}", fill="red")
                        
#                         vis_path = f"debug_face_detection_{path.stem}.jpg"
#                         vis_image.save(vis_path)
#                         print(f"    * Saved detection visualization: {vis_path}")
                
#                 # Step 4: Extract embeddings (this is what gets returned)
#                 embeddings = torch.stack([torch.from_numpy(f.normed_embedding) for f in faces])
                
#                 print(f"[DEBUG] Final concept signal:")
#                 print(f"  - Concept signal shape: {embeddings.shape}")
#                 print(f"  - Concept signal type: {type(embeddings)}")
#                 print(f"  - This contains FACE EMBEDDINGS, not full image embeddings")
                
#                 output[path] = embeddings
        
#         return output

# # Additional debugging function to test the internal InsightFace process
# def debug_insightface_internals():
#     """
#     Test function to understand what InsightFace does internally
#     """
#     print("\n" + "="*80)
#     print("DEBUGGING INSIGHTFACE INTERNAL PROCESS")
#     print("="*80)
    
#     # Initialize InsightFace
#     app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     app.prepare(ctx_id=0, det_size=(640, 640))
    
#     # Load test image
#     image_path = "path/to/your/test/image.jpg"  # Replace with actual path
#     image = Image.open(image_path).convert("RGB")
#     image_np = np.array(image)
    
#     print(f"Input image shape: {image_np.shape}")
    
#     # Get faces
#     faces = app.get(image_np)
    
#     print(f"Number of detected faces: {len(faces)}")
    
#     for i, face in enumerate(faces):
#         print(f"\nFace {i} analysis:")
#         print(f"  Bounding box: {face.bbox}")
#         print(f"  Face embedding shape: {face.normed_embedding.shape}")
        
#         # Manually crop the face to show what InsightFace is working with
#         bbox = face.bbox.astype(int)
#         x1, y1, x2, y2 = bbox
        
#         # This is essentially what InsightFace does internally
#         face_region = image_np[y1:y2, x1:x2]  # Crop face from full image
        
#         print(f"  Face region shape: {face_region.shape}")
#         print(f"  Face region is cropped from coordinates: ({x1},{y1}) to ({x2},{y2})")
        
#         # Save the cropped face
#         face_crop_pil = Image.fromarray(face_region)
#         face_crop_pil.save(f"manual_face_crop_{i}.jpg")
#         print(f"  Saved manual face crop: manual_face_crop_{i}.jpg")
        
#         # The embedding comes from processing this cropped region, not the full image
#         print(f"  The 512-dim embedding represents THIS cropped face, not the full image")

# # Quick test function
# def quick_face_debug_test():
#     """
#     Quick test to confirm face processing
#     """
#     image_path = "path/to/test/image.jpg"  # Replace with actual path
    
#     head = FaceConceptHead(debug=True)
#     result = head.extract_signal([Path(image_path)])
    
#     print(f"\nFINAL RESULT:")
#     for path, signal in result.items():
#         if signal is not None:
#             print(f"{path}: Face embeddings shape {signal.shape}")
#             print(f"These are embeddings of DETECTED FACES, not the full image")
#         else:
#             print(f"{path}: No faces detected")

# # # Usage:
# # # 1. Replace the FaceConceptHead class with the debug version above
# # # 2. Run your inference with a test image
# # # 3. Check the saved debug images to confirm the process