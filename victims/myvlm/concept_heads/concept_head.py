# concept_heads/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from torch import Tensor

class ConceptHead(ABC):
    @abstractmethod
    def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Optional[Tensor]]:
        """
        Defines the signal used to determine if the concept is present in a given image or not.
        Return a dict mapping each input image path to:
          - a Tensor signal (e.g., face embedding(s), logits, etc.), or
          - None if the signal can't be computed (e.g., no face detected).

        Notes:
        - Face-recognition heads typically return L2-normalized embeddings.
        - Object heads may return classifier logits.
        """
        raise NotImplementedError


# from abc import ABC, abstractmethod
# from pathlib import Path
# from torch import Tensor
# from typing import List, Dict


# class ConceptHead(ABC):

#     @abstractmethod
#     def extract_signal(self, image_paths: List[Path]) -> Dict[Path, Tensor]:
#         """
#         Defines the signal used to determine if the concept is present in a given image or not.
#         For faces, this is the face embedding, while for objects this is the logits obtained from our linear classifier.
#         """
#         raise NotImplementedError
