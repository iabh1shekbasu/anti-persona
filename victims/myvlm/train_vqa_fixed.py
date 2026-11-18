"""MyVLM concept training with ONE data-loader fix (used for every MyVLM victim in the paper).

Upstream bug: concept_embedding_training/datasets/llava_dataset.py LLaVADataset._get_target does
`additional_questions += [caption instruction]` on the STORED per-photo Q&A list, so every draw appends one more caption
copy (VQA share 9/10 at step 0 -> ~4% at step 199). This wrapper runs MyVLM's train.py UNCHANGED, with _get_target
wrapped so the original method receives a shallow COPY of that photo's list (restored right after): the caption is
appended to the copy, the stored list stays 9 Q&A, the sampling mix stays 9 VQA : 1 caption. Same RNG draws, same
code otherwise; MyVLM's files are not edited.

    python train_vqa_fixed.py --config_path <training config .yaml>      (run from this folder)
"""
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
MV = str(Path(__file__).resolve().parent)
sys.path.insert(0, MV)
from concept_embedding_training.datasets import llava_dataset as L   # noqa: E402

_orig = L.LLaVADataset._get_target


def _get_target(self, idx, batch):
    data = self.additional_vqa_data
    if data is not None and len(data['image_paths']) > 0:
        q_idx = data['image_paths'].index(str(batch['image_paths']))
        stored = data['questions_and_answers'][q_idx]
        data['questions_and_answers'][q_idx] = list(stored)
        try:
            return _orig(self, idx, batch)
        finally:
            data['questions_and_answers'][q_idx] = stored
    return _orig(self, idx, batch)


L.LLaVADataset._get_target = _get_target
print('VQA_FIX active: LLaVADataset._get_target appends the caption instruction to a copy (stored lists stay 9 Q&A)', flush=True)
sys.argv = [f'{MV}/concept_embedding_training/train.py', *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name='__main__')
