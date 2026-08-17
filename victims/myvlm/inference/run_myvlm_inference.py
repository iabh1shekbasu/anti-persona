import json
from pathlib import Path
from typing import Dict, List

import pyrallis
import torch

from concept_embedding_training.data_utils import EmbeddingData
from concept_heads.clip.head import CLIPConceptHead
from concept_heads.face_recognition.head import FaceConceptHead
from configs.inference_config import InferenceConfig
from inference import inference_utils
from myvlm import myblip2, myllava, myminigpt_v2
from myvlm.common import ConceptType, seed_everything, CLIP_MODEL_NAME, VLMType, VLM_TO_LAYER
from myvlm.myvlm import MyVLM
from vlms.blip2_wrapper import BLIP2Wrapper
from vlms.llava_wrapper import LLaVAWrapper
from vlms.minigpt_wrapper import MiniGPTWrapper

VLM_TYPE_TO_WRAPPER = {
    VLMType.BLIP2: BLIP2Wrapper,
    VLMType.LLAVA: LLaVAWrapper,
    VLMType.MINIGPT_V2: MiniGPTWrapper
}
VLM_TYPE_TO_MYVLM = {
    VLMType.BLIP2: myblip2.MyBLIP2,
    VLMType.LLAVA: myllava.MyLLaVA,
    VLMType.MINIGPT_V2: myminigpt_v2.MyMiniGPT_v2
}


@pyrallis.wrap()
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)

    # Load the concept head that was previously trained
    if cfg.concept_type == ConceptType.OBJECT:
        head_path = cfg.concept_head_path / f'{CLIP_MODEL_NAME}-{cfg.concept_name}-step-{cfg.classifier_step}.pt'
        concept_head = CLIPConceptHead(head_path)
    else:
        concept_head = FaceConceptHead()

    concept_signals = concept_head.extract_signal(cfg.image_paths)
    concept_signals = [concept_signals[path] for path in cfg.image_paths]

    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
    myvlm = VLM_TYPE_TO_MYVLM[cfg.vlm_type](vlm=vlm_wrapper,
                                            layer=VLM_TO_LAYER[cfg.vlm_type],
                                            concept_name=cfg.concept_name,
                                            cfg=cfg)

    iteration_to_concept_data = torch.load(cfg.checkpoint_path /
                                           f'concept_embeddings_{cfg.vlm_type}_{cfg.personalization_task}.pt')
    iterations = cfg.iterations if cfg.iterations is not None else list(iteration_to_concept_data.keys())

    outputs = run_inference(myvlm=myvlm,
                            concept_signals=concept_signals,
                            iterations=iterations,
                            iteration_to_concept_data=iteration_to_concept_data,
                            cfg=cfg)

    # Save results to json file
    with open(cfg.inference_output_path / f'inference_outputs_{cfg.vlm_type}_{cfg.personalization_task}.json', 'w') as f:
        json.dump(outputs, f, indent=4)


def run_inference(myvlm: MyVLM,
                  concept_signals: Dict[str, torch.Tensor],
                  iterations: List[int],
                  iteration_to_concept_data: Dict[str, EmbeddingData],
                  cfg: InferenceConfig) -> Dict[str, Dict]:
    print("*" * 100)
    print("RUNNING INFERENCE")
    print("*" * 100)
    outputs = {}
    for iteration in iterations:
        print('#' * 100)
        print(f"Running on iteration: {iteration}")
        inference_utils.set_concept_embeddings(vlm_wrapper=myvlm.vlm,
                                               concept_embeddings=iteration_to_concept_data[iteration],
                                               iteration=iteration,
                                               cfg=cfg)
        print('-' * 100)
        outputs[f'iteration_{iteration}'] = {}
        for image_idx, image in enumerate(cfg.image_paths):
            outputs[f'iteration_{iteration}'][str(image)] = {}
            for prompt in cfg.prompts:
                prompt = prompt.format(concept=cfg.concept_identifier)  # Add the identifier to prompt, if needed
                inputs = myvlm.vlm.preprocess(image, prompt)
                output = myvlm.vlm.generate(inputs, concept_signals=concept_signals[image_idx])
                outputs[f'iteration_{iteration}'][str(image)][prompt] = output[0]
                print(f"{Path(image).stem} | Input: {prompt} | Output: {output[0]}")

                if cfg.vlm_type == VLMType.MINIGPT_V2 and '[refer]' in prompt:
                    output_path = cfg.inference_output_path / 'rec_results' / \
                                  f'{Path(image).stem}---iteration_{iteration}.jpg'
                    myvlm.vlm.draw_and_save_referring_localization_result(image_tensor=inputs['image'],
                                                                          result_str=output[0],
                                                                          output_path=output_path)

            print('-' * 100)
        print("#" * 100)
    return outputs


if __name__ == '__main__':
    main()


# # Modular

# import json
# import argparse
# from pathlib import Path
# from typing import Dict, List

# import torch

# from concept_embedding_training.data_utils import EmbeddingData
# from concept_heads.clip.head import CLIPConceptHead
# from concept_heads.face_recognition.head import FaceConceptHead
# from inference import inference_utils
# from myvlm import myblip2, myllava, myminigpt_v2
# from myvlm.common import (
#     ConceptType, seed_everything, CLIP_MODEL_NAME,
#     VLMType, VLM_TO_LAYER, VLM_TO_PROMPTS,
#     PersonalizationTask, VALID_IMAGE_EXTENSIONS
# )
# from myvlm.myvlm import MyVLM
# from vlms.blip2_wrapper import BLIP2Wrapper
# from vlms.llava_wrapper import LLaVAWrapper
# from vlms.minigpt_wrapper import MiniGPTWrapper

# VLM_TYPE_TO_WRAPPER = {
#     VLMType.BLIP2:      BLIP2Wrapper,
#     VLMType.LLAVA:      LLaVAWrapper,
#     VLMType.MINIGPT_V2: MiniGPTWrapper,
# }
# VLM_TYPE_TO_MYVLM = {
#     VLMType.BLIP2:      myblip2.MyBLIP2,
#     VLMType.LLAVA:      myllava.MyLLaVA,
#     VLMType.MINIGPT_V2: myminigpt_v2.MyMiniGPT_v2,
# }

# # CLI choices -> enum values (avoids issues with hyphenated VLMType values)
# STR_TO_VLM_TYPE = {
#     'BLIP2':      VLMType.BLIP2,
#     'LLAVA':      VLMType.LLAVA,
#     'MINIGPT_V2': VLMType.MINIGPT_V2,
# }
# STR_TO_TASK = {
#     'CAPTIONING': PersonalizationTask.CAPTIONING,
#     'VQA':        PersonalizationTask.VQA,
#     'REC':        PersonalizationTask.REC,
# }
# STR_TO_CONCEPT_TYPE = {
#     'PERSON': ConceptType.PERSON,
#     'OBJECT': ConceptType.OBJECT,
# }


# def parse_args():
#     parser = argparse.ArgumentParser(description="MyVLM Inference")

#     # --- Identity ---
#     parser.add_argument("--concept_name",         type=str, required=True,
#                         help="e.g. willinvietnam")
#     parser.add_argument("--concept_identifier",   type=str, default=None,
#                         help="Identifier used during training. Defaults to concept_name.")
#     parser.add_argument("--concept_type",         type=str, default="PERSON",
#                         choices=list(STR_TO_CONCEPT_TYPE.keys()))

#     # --- Model / Task ---
#     parser.add_argument("--vlm_type",             type=str, default="LLAVA",
#                         choices=list(STR_TO_VLM_TYPE.keys()))
#     parser.add_argument("--personalization_task", type=str, default="CAPTIONING",
#                         choices=list(STR_TO_TASK.keys()))
#     parser.add_argument("--device",               type=str, default="cuda")
#     parser.add_argument("--torch_dtype",          type=str, default="bfloat16",
#                         choices=["float16", "float32", "bfloat16"])

#     # --- Paths ---
#     parser.add_argument("--image_paths",          type=str, required=True,
#                         help="Path to folder of inference images")
#     parser.add_argument("--checkpoint_path",      type=str, required=True,
#                         help="Root outputs folder (concept_name/seed_X appended if not already pointing to seed_ dir)")
#     parser.add_argument("--output_path",          type=str, default="./outputs",
#                         help="Root folder for saving inference results (default: ./outputs)")
#     parser.add_argument("--concept_head_path",    type=str, default="./object_concept_heads",
#                         help="Root folder for OBJECT concept heads (concept_name/seed_X appended if needed)")
#     parser.add_argument("--classifier_step",      type=int, default=500,
#                         help="Step used for OBJECT concept head filename")

#     # --- Misc ---
#     parser.add_argument("--seed",                 type=int, default=42)
#     parser.add_argument("--iterations",           type=int, nargs="+", default=None,
#                         help="Specific checkpoint iterations to run. Defaults to all saved.")
#     parser.add_argument("--prompts",              type=str, nargs="+", default=None,
#                         help="Override default prompts for the task")

#     return parser.parse_args()


# def build_config(args):
#     """Mirrors the path resolution and defaulting logic of InferenceConfig.__post_init__."""
#     cfg = argparse.Namespace(**vars(args))

#     # Resolve enums using explicit mappings (VLMType has hyphenated values, can't use [] lookup)
#     cfg.concept_type         = STR_TO_CONCEPT_TYPE[args.concept_type]
#     cfg.vlm_type             = STR_TO_VLM_TYPE[args.vlm_type]
#     cfg.personalization_task = STR_TO_TASK[args.personalization_task]

#     # Resolve torch dtype
#     cfg.torch_dtype = {
#         "float16":  torch.float16,
#         "float32":  torch.float32,
#         "bfloat16": torch.bfloat16,
#     }[args.torch_dtype]

#     # Default concept_identifier = concept_name
#     if cfg.concept_identifier is None:
#         cfg.concept_identifier = cfg.concept_name

#     # --- Path resolution (mirrors InferenceConfig.__post_init__) ---
#     seed_suffix       = f"seed_{cfg.seed}"
#     checkpoint_path   = Path(args.checkpoint_path)
#     concept_head_path = Path(args.concept_head_path)

#     if not checkpoint_path.name.startswith("seed_"):
#         checkpoint_path = checkpoint_path / cfg.concept_name / seed_suffix
#     if not concept_head_path.name.startswith("seed_"):
#         concept_head_path = concept_head_path / cfg.concept_name / seed_suffix

#     cfg.checkpoint_path   = checkpoint_path
#     cfg.concept_head_path = concept_head_path

#     # inference_output_path is derived from output_path, not checkpoint_path
#     output_path = Path(args.output_path)
#     cfg.inference_output_path = output_path / cfg.concept_name / "inference_outputs" / seed_suffix
#     cfg.inference_output_path.mkdir(parents=True, exist_ok=True)

#     # Verify checkpoint exists (mirrors _verify_concept_embeddings_exist)
#     if not cfg.checkpoint_path.exists():
#         raise ValueError(f"Concept checkpoint path {cfg.checkpoint_path} does not exist!")

#     # Verify concept head exists for OBJECT (mirrors _verify_concept_heads_exist)
#     if cfg.concept_type == ConceptType.OBJECT and not cfg.concept_head_path.exists():
#         raise ValueError(f"Concept head path {cfg.concept_head_path} does not exist!")

#     # Collect image paths from directory (mirrors __post_init__ image_paths handling)
#     image_dir = Path(args.image_paths)
#     cfg.image_paths = sorted([
#         str(p) for p in image_dir.glob("*")
#         if p.suffix in VALID_IMAGE_EXTENSIONS  # use extensions exactly as defined in common.py
#     ])
#     assert cfg.image_paths, f"No images found in {image_dir}"

#     # Threshold (mirrors InferenceConfig.__post_init__ — 0.55 for PERSON, 0.5 for OBJECT)
#     cfg.threshold = 0.5 if cfg.concept_type == ConceptType.OBJECT else 0.55

#     # Prompts (mirrors InferenceConfig.__post_init__)
#     if cfg.prompts is None:
#         cfg.prompts = VLM_TO_PROMPTS[cfg.vlm_type].get(cfg.personalization_task, None)
#         if cfg.prompts is None:
#             raise ValueError(
#                 f"No default prompts defined for task {args.personalization_task} "
#                 f"with VLM {args.vlm_type}. Pass --prompts explicitly."
#             )

#     return cfg


# def main():
#     args = parse_args()
#     cfg  = build_config(args)

#     seed_everything(cfg.seed)

#     print(f"Checkpoint path : {cfg.checkpoint_path}")
#     print(f"Output path     : {cfg.inference_output_path}")
#     print(f"Images found    : {len(cfg.image_paths)}")
#     print(f"Threshold       : {cfg.threshold}")
#     print(f"Prompts         : {cfg.prompts}")

#     # Load concept head
#     if cfg.concept_type == ConceptType.OBJECT:
#         head_path = cfg.concept_head_path / f"{CLIP_MODEL_NAME}-{cfg.concept_name}-step-{cfg.classifier_step}.pt"
#         concept_head = CLIPConceptHead(head_path)
#     else:
#         concept_head = FaceConceptHead()

#     concept_signals = concept_head.extract_signal(cfg.image_paths)
#     concept_signals = [concept_signals[path] for path in cfg.image_paths]

#     vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
#     myvlm = VLM_TYPE_TO_MYVLM[cfg.vlm_type](
#         vlm=vlm_wrapper,
#         layer=VLM_TO_LAYER[cfg.vlm_type],
#         concept_name=cfg.concept_name,
#         cfg=cfg,
#     )

#     # Checkpoint filename uses enum .value (e.g. 'llava', 'captioning') to match training output
#     ckpt_file = cfg.checkpoint_path / f"concept_embeddings_{cfg.vlm_type.value}_{cfg.personalization_task.value}.pt"
#     iteration_to_concept_data = torch.load(ckpt_file)
#     iterations = cfg.iterations if cfg.iterations is not None else list(iteration_to_concept_data.keys())

#     outputs = run_inference(
#         myvlm=myvlm,
#         concept_signals=concept_signals,
#         iterations=iterations,
#         iteration_to_concept_data=iteration_to_concept_data,
#         cfg=cfg,
#     )

#     out_file = cfg.inference_output_path / f"inference_outputs_{cfg.vlm_type.value}_{cfg.personalization_task.value}.json"
#     with open(out_file, "w") as f:
#         json.dump(outputs, f, indent=4)
#     print(f"\nSaved outputs to {out_file}")


# def run_inference(
#     myvlm: MyVLM,
#     concept_signals: Dict[str, torch.Tensor],
#     iterations: List[int],
#     iteration_to_concept_data: Dict[str, EmbeddingData],
#     cfg,
# ) -> Dict[str, Dict]:
#     print("*" * 100)
#     print("RUNNING INFERENCE")
#     print("*" * 100)
#     outputs = {}
#     for iteration in iterations:
#         print("#" * 100)
#         print(f"Running on iteration: {iteration}")
#         inference_utils.set_concept_embeddings(
#             vlm_wrapper=myvlm.vlm,
#             concept_embeddings=iteration_to_concept_data[iteration],
#             iteration=iteration,
#             cfg=cfg,
#         )
#         print("-" * 100)
#         outputs[f"iteration_{iteration}"] = {}
#         for image_idx, image in enumerate(cfg.image_paths):
#             outputs[f"iteration_{iteration}"][str(image)] = {}
#             for prompt in cfg.prompts:
#                 prompt = prompt.format(concept=cfg.concept_identifier)
#                 inputs = myvlm.vlm.preprocess(image, prompt)
#                 output = myvlm.vlm.generate(inputs, concept_signals=concept_signals[image_idx])
#                 outputs[f"iteration_{iteration}"][str(image)][prompt] = output[0]
#                 print(f"{Path(image).stem} | Input: {prompt} | Output: {output[0]}")

#                 if cfg.vlm_type == VLMType.MINIGPT_V2 and "[refer]" in prompt:
#                     rec_dir = cfg.inference_output_path / "rec_results"
#                     rec_dir.mkdir(parents=True, exist_ok=True)
#                     output_path = rec_dir / f"{Path(image).stem}---iteration_{iteration}.jpg"
#                     myvlm.vlm.draw_and_save_referring_localization_result(
#                         image_tensor=inputs["image"],
#                         result_str=output[0],
#                         output_path=output_path,
#                     )
#             print("-" * 100)
#         print("#" * 100)
#     return outputs


# if __name__ == "__main__":
#     main()