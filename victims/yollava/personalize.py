# #!/usr/bin/env python3
# import os
# import random
# import numpy as np
# import torch
# import argparse
# import json
# import torch.nn as nn
# from llava.eval.my_llava import get_model, get_query
# from llava.mm_utils import (
#     get_model_name_from_path,
#     tokenizer_image_token_batch,
# )
# from sklearn.metrics import precision_recall_fscore_support
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from tqdm import tqdm
# from llava.eval.my_llava import *

# IMAGE_TOKEN_INDEX = -200

# # ─── Worker init for DataLoader ───────────────────────────────────────────────
# def make_worker_init_fn(base_seed):
#     def worker_init_fn(worker_id):
#         seed = base_seed + worker_id
#         import random, numpy as np, torch
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#     return worker_init_fn

# # ─── Arg parsing ──────────────────────────────────────────────────────────────
# def get_train_args():
#     parser = argparse.ArgumentParser()
#     # Model
#     parser.add_argument("--model_path",    type=str,   default="liuhaotian/llava-v1.6-vicuna-13b")
#     parser.add_argument("--model_base",    type=str,   default=None)
#     parser.add_argument("--model_name",    type=str,   default=None)
#     parser.add_argument("--conv_mode",     type=str,   default=None)
#     parser.add_argument("--sep",           type=str,   default=",")
#     parser.add_argument("--temperature",   type=float, default=0.2)
#     parser.add_argument("--top_p",         type=float, default=None)
#     parser.add_argument("--num_beams",     type=int,   default=1)
#     parser.add_argument("--max_new_tokens",type=int,   default=512)
#     # Dataset
#     parser.add_argument("--data_root",     type=str,   default='/nobackup/thao-data/dataset/stuffed-animals')
#     parser.add_argument("--sks_name",      type=str,   default='shiba-yellow')
#     parser.add_argument("--prefix_token",  type=int,   default=4)
#     parser.add_argument("--flip_p",        type=float, default=0.5)
#     parser.add_argument("--train_lm_head", action='store_true')
#     parser.add_argument("--user_prompt",   action='store_true')
#     parser.add_argument("--extreme_negative", action='store_true')
#     parser.add_argument("--recog_only",    action='store_true')
#     parser.add_argument("--random_image",  action='store_true')
#     parser.add_argument("--text_only",     action='store_true')
#     parser.add_argument("--suffix_prompt", type=str,   default=None)
#     # Logging & checkpoints
#     parser.add_argument("--tensorboard_path", type=str, default='./runs/')
#     parser.add_argument("--checkpoint_path",  type=str, default='./checkpoints/')
#     parser.add_argument("--exp_name",         type=str, default='./debug/')
#     parser.add_argument("--log_every",        type=int, default=1)
#     parser.add_argument("--epoch",            type=int, default=20)
#     parser.add_argument("--seed",             type=int, default=42)
#     return parser.parse_args()

# # ─── Main ─────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     args = get_train_args()

#     # ─── SET SEED FROM ARGS ───────────────────────────────────────────────
#     os.environ["PYTHONHASHSEED"] = str(args.seed)
    
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
    
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     # prepare logging & checkpoints
#     writer        = SummaryWriter(os.path.join(args.tensorboard_path, args.sks_name, args.exp_name))
#     save_location = os.path.join(args.checkpoint_path,   args.sks_name, args.exp_name)
#     os.makedirs(save_location, exist_ok=True)
#     args.model_name = get_model_name_from_path(args.model_path)

#     # load model + tokenizer + image processor
#     tokenizer, model, image_processor, context_len = get_model(args)

#     print("Vision tower:",  model.model.vision_tower)
#     print("Projector:",      model.model.mm_projector)

#     # ─── Datasets & DataLoaders ─────────────────────────────────────────────────
#     train_dataset = PersonalizedDataset_Mixture(
#         data_root       = args.data_root,
#         sks_name        = args.sks_name,
#         tokenizer       = tokenizer,
#         config          = model.config,
#         image_processor = image_processor,
#         device          = model.device,
#         flip_p          = args.flip_p,
#         train_lm_head   = args.train_lm_head,
#         extreme_negative= args.extreme_negative,
#         recog_only      = args.recog_only,
#         random_image    = args.random_image,
#         text_only       = args.text_only,
#     )

#     # fixed‐seed shuffle generator
#     g = torch.Generator()
#     g.manual_seed(args.seed)

#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size     = 1,
#         shuffle        = True,
#         generator      = g,
#         num_workers    = 1,
#         worker_init_fn = make_worker_init_fn(args.seed),
#     )

#     test_dataset = PersonalizedDataset(
#         data_root       = args.data_root,
#         sks_name        = args.sks_name,
#         train_image_paths= train_dataset.images_path,
#         tokenizer       = tokenizer,
#         config          = model.config,
#         image_processor = image_processor,
#         device          = model.device,
#         set             = 'test',
#     )

#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size  = 1,
#         shuffle     = False,
#         num_workers = 4,
#     )

#     print('sks is: ', args.sks_name)
#     print('Number of training samples:', len(train_dataset))

#     # ─── Prepare placeholder tokens & optimizer ─────────────────────────────────
#     if args.prefix_token > 0:
#         prefix_tokens     = [f'<token{i}>' for i in range(args.prefix_token)]
#         placeholder_tokens= [f'<{args.sks_name}>'] + prefix_tokens
#         if args.suffix_prompt:
#             sks_prompt = f"{placeholder_tokens[0]} {args.suffix_prompt}".replace('<sks>', f'<{args.sks_name}>')
#         else:
#             sks_prompt = f"{placeholder_tokens[0]} is {''.join(prefix_tokens)}."
#     else:
#         placeholder_tokens = [f'<{args.sks_name}>']
#         sks_prompt         = placeholder_tokens[0]

#     print('system prompt will add:', sks_prompt)
#     num_added_tokens      = tokenizer.add_tokens(placeholder_tokens)
#     placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

#     model.resize_token_embeddings(len(tokenizer))

#     orig_embeds_params = model.get_input_embeddings().weight.data.clone()
#     orig_lm_params     = model.lm_head.weight.data.clone()

#     trainable_params = [model.get_input_embeddings().weight, model.lm_head.weight]
#     optimizer = torch.optim.AdamW(trainable_params,
#                                   lr=1e-3,
#                                   betas=(0.9,0.999),
#                                   weight_decay=1e-2,
#                                   eps=1e-8)

#     model.train()
#     model.model.requires_grad_(False)
#     model.get_input_embeddings().weight.requires_grad_(True)

#     best_acc = 0.0

#     for epoch in range(args.epoch):
#         # print trainable params
#         for name, p in model.named_parameters():
#             if p.requires_grad:
#                 print(name, "requires_grad")

#         # ─── Training loop ────────────────────────────────────────────────
#         for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
#             optimizer.zero_grad()

#             # build prompt
#             if args.user_prompt:
#                 prompts = [ get_query(args, sks_prompt + ' ' + q, model=model, sks_system_prompt=None)
#                             .conv.get_prompt()
#                             for q in batch['query'] ]
#             else:
#                 prompts = [ get_query(args, q, model=model, sks_system_prompt=sks_prompt)
#                             .conv.get_prompt()
#                             for q in batch['query'] ]

#             prompts = [ p + ' ' + a for p,a in zip(prompts, batch['answer']) ]
#             if not batch['has_image']:
#                 prompts = [ p.replace('<image>\n','') for p in prompts ]

#             input_ids, labels = tokenizer_image_token_batch(
#                 prompts,
#                 tokenizer,
#                 IMAGE_TOKEN_INDEX,
#                 return_tensors="pt",
#                 return_labels=True,
#             )
#             input_ids = input_ids.cuda()
#             labels    = labels.cuda()

#             with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
#                 if not batch['has_image']:
#                     outputs = model(input_ids, labels=labels)
#                 else:
#                     outputs = model(input_ids,
#                                     images=batch['images'][0],
#                                     labels=labels,
#                                     image_sizes=batch['image_sizes'])
#             loss = outputs.loss
#             loss.backward()

#             # ─── [CRITICAL FIX] Zero gradients for frozen tokens BEFORE step ───
#             # This prevents AdamW from accumulating momentum on frozen weights
#             if model.get_input_embeddings().weight.grad is not None:
#                 idx_no_upd = torch.ones((len(tokenizer),), dtype=torch.bool)
#                 idx_no_upd[placeholder_token_ids] = False
#                 model.get_input_embeddings().weight.grad[idx_no_upd] = 0
#                 if args.train_lm_head and model.lm_head.weight.grad is not None:
#                      model.lm_head.weight.grad[idx_no_upd] = 0
#             # ──────────────────────────────────────────────────────────────────

#             optimizer.step()

#             # lock down embeddings outside placeholder (Reset weights data)
#             index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool)
#             index_no_updates[placeholder_token_ids] = False
#             with torch.no_grad():
#                 model.get_input_embeddings().weight.data[index_no_updates] = orig_embeds_params[index_no_updates]
#                 model.lm_head.weight.data[index_no_updates]            = orig_lm_params[index_no_updates]

#             # log scalars
#             global_step = epoch * len(train_dataloader) + step
#             writer.add_scalar('Loss/Train', loss.item(), global_step)
#             writer.add_scalar('Loss/Token-Norm',
#                               model.get_input_embeddings().weight[placeholder_token_ids].norm().item(),
#                               global_step)

#         # save checkpoint
#         if epoch % args.log_every == 0:
#             print('Saving checkpoints…')
#             torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids],
#                        os.path.join(save_location, f'{epoch}-token.pt'))
#             torch.save(model.lm_head.weight.data[placeholder_token_ids],
#                        os.path.join(save_location, f'{epoch}-lmhead.pt'))

#         # ─── Evaluation ───────────────────────────────────────────────────
#         with torch.no_grad():
#             print('Running evaluation…')
#             preds, gts = [], []
#             for batch in tqdm(test_dataloader, desc="Eval"):
#                 if args.user_prompt:
#                     prompts = [ get_query(args, sks_prompt + ' ' + q, model=model, sks_system_prompt=None)
#                                 .conv.get_prompt()
#                                 for q in batch['query'] ]
#                 else:
#                     prompts = [ get_query(args, q, model=model, sks_system_prompt=sks_prompt)
#                                 .conv.get_prompt()
#                                 for q in batch['query'] ]

#                 input_ids, _ = tokenizer_image_token_batch(
#                     prompts,
#                     tokenizer,
#                     IMAGE_TOKEN_INDEX,
#                     return_tensors="pt",
#                     return_labels=False,
#                 )
#                 outputs = model.generate(
#                     input_ids.cuda(),
#                     images=batch['images'][0].cuda(),
#                     image_sizes=batch['image_sizes'],
#                     do_sample=False,
#                     num_beams=1,
#                 )
#                 answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#                 preds.append(answer)
#                 gts.append(batch['answer'][0])

#             preds = np.array(preds)
#             gts   = np.array(gts)
#             idx_yes = np.where(gts=='Yes')[0]
#             idx_no  = np.where(gts=='No')[0]
#             # Handle empty arrays to avoid div by zero
#             acc_yes = (preds[idx_yes]=='Yes').sum()/len(idx_yes) if len(idx_yes) > 0 else 0
#             acc_no  = (preds[idx_no]=='No').sum()/len(idx_no) if len(idx_no) > 0 else 0
#             avg_acc = (acc_yes + acc_no)/2

#             writer.add_scalar('Accuracy/sks', acc_yes, epoch)
#             writer.add_scalar('Accuracy/no-sks', acc_no, epoch)
#             writer.add_scalar('Accuracy/ave', avg_acc, epoch)

#             if avg_acc > best_acc and epoch > 4:
#                 best_acc = avg_acc
#                 print(f"New best avg acc={best_acc:.4f}, saving best-token+lmhead…")
#                 torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids],
#                            os.path.join(save_location, 'best-token.pt'))
#                 torch.save(model.lm_head.weight.data[placeholder_token_ids],
#                            os.path.join(save_location, 'best-lmhead.pt'))


#!/usr/bin/env python3
import os
# ─── Basic seed only (no determinism) ─────────────────────────────────────────
# ─── Practical deterministic setup ────────────────────────────────────────────
# os.environ["PYTHONHASHSEED"] = "42"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA >= 10.2

import random
import numpy as np
import torch

# Seed core RNGs only - no deterministic algorithms
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

# Enable CuDNN determinism (this usually works)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ─── Standard imports & your LLAVA setup ──────────────────────────────────────
import argparse
import json
import torch.nn as nn
from llava.eval.my_llava import get_model, get_query
from llava.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token_batch,
)
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from llava.eval.my_llava import *

IMAGE_TOKEN_INDEX = -200

# ─── Worker init for DataLoader ───────────────────────────────────────────────
def make_worker_init_fn(base_seed):
    def worker_init_fn(worker_id):
        seed = base_seed + worker_id
        import random, numpy as np, torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return worker_init_fn

# ─── Arg parsing ──────────────────────────────────────────────────────────────
def get_train_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_path",    type=str,   default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--model_base",    type=str,   default=None)
    parser.add_argument("--model_name",    type=str,   default=None)
    parser.add_argument("--conv_mode",     type=str,   default=None)
    parser.add_argument("--sep",           type=str,   default=",")
    parser.add_argument("--temperature",   type=float, default=0.2)
    parser.add_argument("--top_p",         type=float, default=None)
    parser.add_argument("--num_beams",     type=int,   default=1)
    parser.add_argument("--max_new_tokens",type=int,   default=512)
    # Dataset
    parser.add_argument("--data_root",     type=str,   default='/nobackup/thao-data/dataset/stuffed-animals')
    parser.add_argument("--sks_name",      type=str,   default='shiba-yellow')
    parser.add_argument("--prefix_token",  type=int,   default=4)
    parser.add_argument("--flip_p",        type=float, default=0.5)
    parser.add_argument("--train_lm_head", action='store_true')
    parser.add_argument("--user_prompt",   action='store_true')
    parser.add_argument("--extreme_negative", action='store_true')
    parser.add_argument("--recog_only",    action='store_true')
    parser.add_argument("--random_image",  action='store_true')
    parser.add_argument("--text_only",     action='store_true')
    parser.add_argument("--suffix_prompt", type=str,   default=None)
    # Logging & checkpoints
    parser.add_argument("--tensorboard_path", type=str, default='./runs/')
    parser.add_argument("--checkpoint_path",  type=str, default='./checkpoints/')
    parser.add_argument("--exp_name",         type=str, default='./debug/')
    parser.add_argument("--log_every",        type=int, default=1)
    parser.add_argument("--epoch",            type=int, default=20)
    parser.add_argument("--seed",             type=int, default=42)  # ← ADD THIS LINE
    return parser.parse_args()

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_train_args()

    # ─── SET SEED FROM ARGS ───────────────────────────────────────────────
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare logging & checkpoints
    writer        = SummaryWriter(os.path.join(args.tensorboard_path, args.sks_name, args.exp_name))
    save_location = os.path.join(args.checkpoint_path,   args.sks_name, args.exp_name)
    os.makedirs(save_location, exist_ok=True)
    args.model_name = get_model_name_from_path(args.model_path)

    # load model + tokenizer + image processor
    tokenizer, model, image_processor, context_len = get_model(args)

    print("Vision tower:",  model.model.vision_tower)
    print("Vision config:", model.model.vision_tower.config)
    print("Hidden size:",    model.model.vision_tower.config.hidden_size)
    print("Projector:",      model.model.mm_projector)

    # attach detailed hooks if you want
    def detailed_hook(module, input, output):
        print(f"\n{module.__class__.__name__}:")
        print(f"  Input shape:  {input[0].shape if isinstance(input, tuple) else input.shape}")
        print(f"  Output shape: {output.shape}")

    model.model.vision_tower._forward_hooks.clear()
    model.model.mm_projector._forward_hooks.clear()
    model.model.vision_tower.register_forward_hook(detailed_hook)
    model.model.mm_projector.register_forward_hook(detailed_hook)

    # ─── Datasets & DataLoaders ─────────────────────────────────────────────────
    train_dataset = PersonalizedDataset_Mixture(
        data_root       = args.data_root,
        sks_name        = args.sks_name,
        tokenizer       = tokenizer,
        config          = model.config,
        image_processor = image_processor,
        device          = model.device,
        flip_p          = args.flip_p,
        train_lm_head   = args.train_lm_head,
        extreme_negative= args.extreme_negative,
        recog_only      = args.recog_only,
        random_image    = args.random_image,
        text_only       = args.text_only,
    )

    # fixed‐seed shuffle generator
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size     = 1,
        shuffle        = True,
        generator      = g,
        num_workers    = 1,
        worker_init_fn = make_worker_init_fn(args.seed),
    )

    test_dataset = PersonalizedDataset(
        data_root       = args.data_root,
        sks_name        = args.sks_name,
        train_image_paths= train_dataset.images_path,
        tokenizer       = tokenizer,
        config          = model.config,
        image_processor = image_processor,
        device          = model.device,
        set             = 'test',
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 4,
    )

    print('sks is: ', args.sks_name)
    print('Number of training samples:', len(train_dataset))

    # ─── Prepare placeholder tokens & optimizer ─────────────────────────────────
    if args.prefix_token > 0:
        prefix_tokens     = [f'<token{i}>' for i in range(args.prefix_token)]
        placeholder_tokens= [f'<{args.sks_name}>'] + prefix_tokens
        if args.suffix_prompt:
            sks_prompt = f"{placeholder_tokens[0]} {args.suffix_prompt}".replace('<sks>', f'<{args.sks_name}>')
        else:
            sks_prompt = f"{placeholder_tokens[0]} is {''.join(prefix_tokens)}."
    else:
        placeholder_tokens = [f'<{args.sks_name}>']
        sks_prompt         = placeholder_tokens[0]

    print('system prompt will add:', sks_prompt)
    num_added_tokens      = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    model.resize_token_embeddings(len(tokenizer))

    orig_embeds_params = model.get_input_embeddings().weight.data.clone()
    orig_lm_params     = model.lm_head.weight.data.clone()

    trainable_params = [model.get_input_embeddings().weight, model.lm_head.weight]
    optimizer = torch.optim.AdamW(trainable_params,
                                  lr=1e-3,
                                  betas=(0.9,0.999),
                                  weight_decay=1e-2,
                                  eps=1e-8)

    model.train()
    model.model.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)

    best_acc = 0.0

    for epoch in range(args.epoch):
        # print trainable params
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name, "requires_grad")

        # ─── Training loop ────────────────────────────────────────────────
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()

            # build prompt
            if args.user_prompt:
                prompts = [ get_query(args, sks_prompt + ' ' + q, model=model, sks_system_prompt=None)
                            .conv.get_prompt()
                            for q in batch['query'] ]
            else:
                prompts = [ get_query(args, q, model=model, sks_system_prompt=sks_prompt)
                            .conv.get_prompt()
                            for q in batch['query'] ]

            prompts = [ p + ' ' + a for p,a in zip(prompts, batch['answer']) ]
            if not batch['has_image']:
                prompts = [ p.replace('<image>\n','') for p in prompts ]

            input_ids, labels = tokenizer_image_token_batch(
                prompts,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
                return_labels=True,
            )
            input_ids = input_ids.cuda()
            labels    = labels.cuda()

            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                if not batch['has_image']:
                    outputs = model(input_ids, labels=labels)
                else:
                    outputs = model(input_ids,
                                    images=batch['images'][0],
                                    labels=labels,
                                    image_sizes=batch['image_sizes'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # lock down embeddings outside placeholder
            index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool)
            index_no_updates[placeholder_token_ids] = False
            with torch.no_grad():
                model.get_input_embeddings().weight.data[index_no_updates] = orig_embeds_params[index_no_updates]
                model.lm_head.weight.data[index_no_updates]            = orig_lm_params[index_no_updates]

            # log scalars
            global_step = epoch * len(train_dataloader) + step
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            writer.add_scalar('Loss/Token-Norm',
                              model.get_input_embeddings().weight[placeholder_token_ids].norm().item(),
                              global_step)

        # save checkpoint
        if epoch % args.log_every == 0:
            print('Saving checkpoints…')
            torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids],
                       os.path.join(save_location, f'{epoch}-token.pt'))
            torch.save(model.lm_head.weight.data[placeholder_token_ids],
                       os.path.join(save_location, f'{epoch}-lmhead.pt'))

        # ─── Evaluation ───────────────────────────────────────────────────
        with torch.no_grad():
            print('Running evaluation…')
            preds, gts = [], []
            for batch in tqdm(test_dataloader, desc="Eval"):
                if args.user_prompt:
                    prompts = [ get_query(args, sks_prompt + ' ' + q, model=model, sks_system_prompt=None)
                                .conv.get_prompt()
                                for q in batch['query'] ]
                else:
                    prompts = [ get_query(args, q, model=model, sks_system_prompt=sks_prompt)
                                .conv.get_prompt()
                                for q in batch['query'] ]

                input_ids, _ = tokenizer_image_token_batch(
                    prompts,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                    return_labels=False,
                )
                outputs = model.generate(
                    input_ids.cuda(),
                    images=batch['images'][0].cuda(),
                    image_sizes=batch['image_sizes'],
                    do_sample=False,
                    num_beams=1,
                )
                answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                preds.append(answer)
                gts.append(batch['answer'][0])

            preds = np.array(preds)
            gts   = np.array(gts)
            idx_yes = np.where(gts=='Yes')[0]
            idx_no  = np.where(gts=='No')[0]
            acc_yes = (preds[idx_yes]=='Yes').sum()/len(idx_yes)
            acc_no  = (preds[idx_no]=='No').sum()/len(idx_no)
            avg_acc = (acc_yes + acc_no)/2

            writer.add_scalar('Accuracy/sks', acc_yes, epoch)
            writer.add_scalar('Accuracy/no-sks', acc_no, epoch)
            writer.add_scalar('Accuracy/ave', avg_acc, epoch)

            if avg_acc > best_acc and epoch > 4:
                best_acc = avg_acc
                print(f"New best avg acc={best_acc:.4f}, saving best-token+lmhead…")
                torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids],
                           os.path.join(save_location, 'best-token.pt'))
                torch.save(model.lm_head.weight.data[placeholder_token_ids],
                           os.path.join(save_location, 'best-lmhead.pt'))