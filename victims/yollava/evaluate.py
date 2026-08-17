# # orig_embeds_params = model.get_input_embeddings().weight.data.clone()
# import argparse
# import glob
# import os
# import random
# import numpy as np

# import torch
# from llava.eval.my_llava import *
# from llava.mm_utils import (get_model_name_from_path, tokenizer_image_token,
#                             tokenizer_image_token_batch)
# from llava.model.builder import load_pretrained_model
# from tqdm import tqdm

# def set_seed(seed=42):
#     # Python hash seed
#     os.environ['PYTHONHASHSEED'] = str(seed)
    
#     # Python random
#     random.seed(seed)
    
#     # Numpy random
#     np.random.seed(seed)
    
#     # PyTorch random
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
    
#     # CUDNN settings
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def worker_init_fn(worker_id):
#     np.random.seed(42 + worker_id)

# # Call at the very beginning
# set_seed(42)


# def get_args():
#     parser = argparse.ArgumentParser()
#     #--- Model related
#     parser.add_argument("--model_path", type=str, default="./llava_ckpts/llava-v1.6-internal-vicuna-13b-336px")
#     parser.add_argument("--model_base", type=str, default=None)
#     parser.add_argument("--model_name", type=str, default=None)
#     parser.add_argument("--conv_mode", type=str, default=None)

#     parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
#     parser.add_argument("--epoch", type=str, default='2')
#     parser.add_argument("--data_root", type=str, default='./yollava-data/test/')
#     parser.add_argument("--sks_name", type=str, default='shiba-yellow')
#     parser.add_argument("--stage", type=str, default='s2')

#     parser.add_argument("--temperature", type=float, default=0.2)
#     parser.add_argument("--top_p", default=None)
#     parser.add_argument("--num_beams", type=int, default=1)
#     parser.add_argument("--max_new_tokens", type=int, default=512)
#     parser.add_argument("--prefix_token", type=int, default=4)
#     #--- Log related
#     parser.add_argument("--exp_name", type=str, default='multi-token')
#     parser.add_argument("--save_txt", action='store_true', default=False)
#     parser.add_argument("--system_prompt", default=False, action='store_true')
#     parser.add_argument("--suffix_prompt", type=str, default=None)

#     return parser.parse_args()

# if __name__ == "__main__":
#     args = get_args()
#     print("SKS Name received from CLI:", args.sks_name)
    
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path=args.model_path,
#         model_base=None,
#         model_name=get_model_name_from_path(args.model_path)
#     )
    
#     prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
#     placeholder_tokens = [f'<{args.sks_name}>']
#     placeholder_tokens.extend(prefix_tokens)
    
#     num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
#     placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
#     # Resize the token embeddings as we are adding new special tokens to the tokenizer
#     model.resize_token_embeddings(len(tokenizer))

#     # Load the token and lm_head embeddings
#     if args.exp_name:
#         token_path = f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-token.pt'
#         lmhead_path = f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-lmhead.pt'
#     else:
#         token_path = f'{args.checkpoint_path}/{args.sks_name}/{args.epoch}-token.pt'
#         lmhead_path = f'{args.checkpoint_path}/{args.sks_name}/{args.epoch}-lmhead.pt'
    
#     sks_token = torch.load(token_path).detach()
#     lm_head = torch.load(lmhead_path).detach()
#     model.get_input_embeddings().weight.requires_grad = False
#     model.lm_head.weight.requires_grad = False
#     model.get_input_embeddings().weight[placeholder_token_ids] = sks_token.to(model.device, dtype=model.dtype)
#     model.lm_head.weight[placeholder_token_ids] = lm_head.detach().to(model.lm_head.weight.device, dtype=model.dtype)
#     print('New tokens are loaded into: ', placeholder_token_ids)

#     # sks_prompt = f"{placeholder_tokens[0]} is {' '.join(placeholder_tokens[1:])}."
#     if args.prefix_token > 0:
#         prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
#         placeholder_tokens = [f'<{args.sks_name}>']
#         placeholder_tokens.extend(prefix_tokens)
#         if args.suffix_prompt is not None:
#             sks_prompt = f"{placeholder_tokens[0]} {args.suffix_prompt}"
#         else:
#             sks_prompt = f"{placeholder_tokens[0]} is {''.join(placeholder_tokens[1:])}"
#         print('system prompt will add:', sks_prompt)
#     else:
#         placeholder_tokens = [f'<{args.sks_name}>']
#         sks_prompt = placeholder_tokens[0]
#         print('system prompt will add:', sks_prompt)
        
#     print('Learned prompt: ', sks_prompt)
#     if args.system_prompt:
#         args = get_query(args, f"Is <{args.sks_name}> in this photo? Answer with a single word or phrase.", model=model, sks_system_prompt=sks_prompt)
#     else:
#         args = get_query(args, sks_prompt + f" Can you see <{args.sks_name}> in this photo? Answer with a single word or phrase.", model=model, sks_system_prompt=None)
    
#     categories = os.listdir(args.data_root)
#     if 'cc12m_images' in args.data_root:
#         categories = [args.sks_name]
#     if '.DS_Store' in categories:
#         categories.remove('.DS_Store')
        
#     os.makedirs(f"./quantitative/{args.sks_name}", exist_ok=True)
#     print('Categories: ')
#     if args.save_txt:
#         for category in categories:
#             with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
#                 f.write(f'{category}\n')
    
#     if args.save_txt:
#         with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
#             f.write(f'Results for {args.sks_name} with epoch {args.epoch} and setting {args.exp_name}\n')
#         print('Results will be saved in: ', f"./quantitative/{args.sks_name}/acc.txt")
    
#     print('✦ . 　⁺ 　 . ✦ . 　⁺ 　 . ✦ Accuracy by category: ')
    
#     # Variables to track combined accuracies
#     positive_correct = 0
#     positive_total = 0
#     negative_correct = 0
#     negative_total = 0
    
#     for category in categories:
#         list_imgs = []
#         for ext in ['jpg', 'jpeg', 'png', "JPG", "JPEG", "PNG"]:
#             list_imgs.extend(glob.glob(os.path.join(args.data_root, category, f'*.{ext}')))
        
#         print(f'\nProcessing {category}: {len(list_imgs)} images found')
        
#         pred = []
#         skipped = 0
        
#         for image_file in list_imgs:
#             try:
#                 images_tensor, image_sizes = get_image_tensor(args, [image_file], model, image_processor)
#                 output, pred_ids = eval_model(args,
#                                     model=model,
#                                     images_tensor=images_tensor,
#                                     image_sizes=image_sizes,
#                                     image_processor=image_processor,
#                                     tokenizer=tokenizer,
#                                     return_ids=True)
                
#                 # Extract Yes/No robustly
#                 output_lower = output.lower().strip()
#                 if 'yes' in output_lower and 'no' not in output_lower:
#                     prediction = 'Yes'
#                 elif 'no' in output_lower and 'yes' not in output_lower:
#                     prediction = 'No'
#                 else:
#                     print(f"  Ambiguous output for {os.path.basename(image_file)}: '{output}' - SKIPPED")
#                     skipped += 1
#                     continue
                    
#                 pred.append(prediction)
                
#             except Exception as e:
#                 print(f"  Error processing {os.path.basename(image_file)}: {e}")
#                 skipped += 1
#                 continue
        
#         if skipped > 0:
#             print(f"  Skipped {skipped}/{len(list_imgs)} images")
        
#         # Skip category if no valid predictions
#         if len(pred) == 0:
#             print(f'{category}: No valid predictions')
#             if args.save_txt:
#                 with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
#                     f.write(f'{category}: nan\n')
#             continue
        
#         # Determine ground truth
#         if category == args.sks_name:
#             if 'laion' in args.data_root:
#                 gt = ['No'] * len(pred)
#             else:
#                 gt = ['Yes'] * len(pred)
#             # This is the positive class (target subject)
#             positive_total += len(gt)
#             positive_correct += (np.array(pred) == np.array(gt)).sum()
#         else:
#             gt = ['No'] * len(pred)
#             # This is negative class (other subjects)
#             negative_total += len(gt)
#             negative_correct += (np.array(pred) == np.array(gt)).sum()
            
#         true_pos = np.array(pred) == np.array(gt)
#         acc = true_pos.sum() / len(gt)
        
#         print(f'GT: {gt}')
#         print(f'Pred: {pred}')
#         print(f'{category}: {acc:.3f} ({true_pos.sum()}/{len(gt)})')
        
#         if args.save_txt:
#             with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
#                 f.write(f'{acc}\n')
    
#     # Calculate and display combined accuracies
#     print('\n' + '='*50)
#     print('COMBINED ACCURACY RESULTS')
#     print('='*50)
    
#     if positive_total > 0:
#         positive_accuracy = positive_correct / positive_total
#         print(f'Positive accuracy ({args.sks_name}): {positive_accuracy:.3f} ({positive_correct}/{positive_total})')
#     else:
#         print(f'Positive accuracy ({args.sks_name}): No images found')
#         positive_accuracy = 0.0
    
#     if negative_total > 0:
#         negative_accuracy = negative_correct / negative_total
#         print(f'Negative accuracy (all others): {negative_accuracy:.3f} ({negative_correct}/{negative_total})')
#     else:
#         print(f'Negative accuracy (all others): No images found')
#         negative_accuracy = 0.0
    
#     print('='*50)
    
#     # Save combined results
#     if args.save_txt:
#         with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
#             f.write(f'\nCOMBINED RESULTS:\n')
#             f.write(f'Positive accuracy ({args.sks_name}): {positive_accuracy:.3f}\n')
#             f.write(f'Negative accuracy (all others): {negative_accuracy:.3f}\n')




# orig_embeds_params = model.get_input_embeddings().weight.data.clone()
import argparse
import glob
import os
import random
import numpy as np

import torch
from llava.eval.my_llava import *
from llava.mm_utils import (get_model_name_from_path, tokenizer_image_token,
                            tokenizer_image_token_batch)
from llava.model.builder import load_pretrained_model
from tqdm import tqdm

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

set_seed(42)


def get_args():
    parser = argparse.ArgumentParser()
    #--- Model related
    parser.add_argument("--model_path", type=str, default="./llava_ckpts/llava-v1.6-internal-vicuna-13b-336px")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)

    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
    parser.add_argument("--epoch", type=str, default='2')
    parser.add_argument("--data_root", type=str, default='./yollava-data/test/')
    parser.add_argument("--sks_name", type=str, default='shiba-yellow')
    parser.add_argument("--stage", type=str, default='s2')

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prefix_token", type=int, default=4)
    #--- Log related
    parser.add_argument("--exp_name", type=str, default='multi-token')
    parser.add_argument("--save_txt", action='store_true', default=False)
    parser.add_argument("--system_prompt", default=False, action='store_true')
    parser.add_argument("--suffix_prompt", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print("SKS Name received from CLI:", args.sks_name)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path)
    )
    
    prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
    placeholder_tokens = [f'<{args.sks_name}>']
    placeholder_tokens.extend(prefix_tokens)
    
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if args.exp_name:
        token_path = f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-token.pt'
        lmhead_path = f'{args.checkpoint_path}/{args.sks_name}/{args.exp_name}/{args.epoch}-lmhead.pt'
    else:
        token_path = f'{args.checkpoint_path}/{args.sks_name}/{args.epoch}-token.pt'
        lmhead_path = f'{args.checkpoint_path}/{args.sks_name}/{args.epoch}-lmhead.pt'
    
    sks_token = torch.load(token_path).detach()
    lm_head = torch.load(lmhead_path).detach()
    model.get_input_embeddings().weight.requires_grad = False
    model.lm_head.weight.requires_grad = False
    model.get_input_embeddings().weight[placeholder_token_ids] = sks_token.to(model.device, dtype=model.dtype)
    model.lm_head.weight[placeholder_token_ids] = lm_head.detach().to(model.lm_head.weight.device, dtype=model.dtype)
    print('New tokens are loaded into: ', placeholder_token_ids)

    if args.prefix_token > 0:
        prefix_tokens = [f'<token{i}>' for i in range(args.prefix_token)]
        placeholder_tokens = [f'<{args.sks_name}>']
        placeholder_tokens.extend(prefix_tokens)
        if args.suffix_prompt is not None:
            sks_prompt = f"{placeholder_tokens[0]} {args.suffix_prompt}"
        else:
            sks_prompt = f"{placeholder_tokens[0]} is {''.join(placeholder_tokens[1:])}"
        print('system prompt will add:', sks_prompt)
    else:
        placeholder_tokens = [f'<{args.sks_name}>']
        sks_prompt = placeholder_tokens[0]
        print('system prompt will add:', sks_prompt)
        
    print('Learned prompt: ', sks_prompt)
    if args.system_prompt:
        args = get_query(args, f"Is <{args.sks_name}> in this photo? Answer with a single word or phrase.", model=model, sks_system_prompt=sks_prompt)
    else:
        args = get_query(args, sks_prompt + f" Can you see <{args.sks_name}> in this photo? Answer with a single word or phrase.", model=model, sks_system_prompt=None)
    
    categories = os.listdir(args.data_root)
    if 'cc12m_images' in args.data_root:
        categories = [args.sks_name]
    if '.DS_Store' in categories:
        categories.remove('.DS_Store')
        
    os.makedirs(f"./quantitative/{args.sks_name}", exist_ok=True)
    print('Categories: ')
    if args.save_txt:
        for category in categories:
            with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
                f.write(f'{category}\n')
    
    if args.save_txt:
        with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
            f.write(f'Results for {args.sks_name} with epoch {args.epoch} and setting {args.exp_name}\n')
        print('Results will be saved in: ', f"./quantitative/{args.sks_name}/acc.txt")
    
    print('✦ . 　⁺ 　 . ✦ . 　⁺ 　 . ✦ Accuracy by category: ')
    
    positive_correct = 0
    positive_total = 0
    negative_correct = 0
    negative_total = 0
    
    for category in categories:
        list_imgs = []
        for ext in ['jpg', 'jpeg', 'png', "JPG", "JPEG", "PNG"]:
            list_imgs.extend(glob.glob(os.path.join(args.data_root, category, f'*.{ext}')))
        
        print(f'\nProcessing {category}: {len(list_imgs)} images found')
        
        pred = []
        valid_image_files = []
        skipped = 0
        
        for image_file in list_imgs:
            try:
                images_tensor, image_sizes = get_image_tensor(args, [image_file], model, image_processor)
                output, pred_ids = eval_model(args,
                                    model=model,
                                    images_tensor=images_tensor,
                                    image_sizes=image_sizes,
                                    image_processor=image_processor,
                                    tokenizer=tokenizer,
                                    return_ids=True)
                
                output_lower = output.lower().strip()
                if 'yes' in output_lower and 'no' not in output_lower:
                    prediction = 'Yes'
                elif 'no' in output_lower and 'yes' not in output_lower:
                    prediction = 'No'
                else:
                    print(f"  Ambiguous output for {os.path.basename(image_file)}: '{output}' - SKIPPED")
                    skipped += 1
                    continue
                    
                pred.append(prediction)
                valid_image_files.append(image_file)
                
            except Exception as e:
                print(f"  Error processing {os.path.basename(image_file)}: {e}")
                skipped += 1
                continue
        
        if skipped > 0:
            print(f"  Skipped {skipped}/{len(list_imgs)} images")
        
        if len(pred) == 0:
            print(f'{category}: No valid predictions')
            if args.save_txt:
                with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
                    f.write(f'{category}: nan\n')
            continue
        
        if category == args.sks_name:
            if 'laion' in args.data_root:
                gt = ['No'] * len(pred)
            else:
                gt = ['Yes'] * len(pred)
            positive_total += len(gt)
            positive_correct += (np.array(pred) == np.array(gt)).sum()
        else:
            gt = ['No'] * len(pred)
            negative_total += len(gt)
            negative_correct += (np.array(pred) == np.array(gt)).sum()
            
        true_pos = np.array(pred) == np.array(gt)
        acc = true_pos.sum() / len(gt)
        
        print(f'GT: {gt}')
        print(f'Pred: {pred}')
        print(f'{category}: {acc:.3f} ({true_pos.sum()}/{len(gt)})')
        
        # Print misclassified images
        wrong_indices = np.where(~true_pos)[0]
        if len(wrong_indices) > 0:
            print(f'  WRONG:')
            for idx in wrong_indices:
                print(f'    {os.path.basename(valid_image_files[idx])}  (GT={gt[idx]}, Pred={pred[idx]})')
        
        if args.save_txt:
            with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
                f.write(f'{acc}\n')
    
    # Calculate and display combined accuracies
    print('\n' + '='*50)
    print('COMBINED ACCURACY RESULTS')
    print('='*50)
    
    if positive_total > 0:
        positive_accuracy = positive_correct / positive_total
        print(f'Positive accuracy ({args.sks_name}): {positive_accuracy:.3f} ({positive_correct}/{positive_total})')
    else:
        print(f'Positive accuracy ({args.sks_name}): No images found')
        positive_accuracy = 0.0
    
    if negative_total > 0:
        negative_accuracy = negative_correct / negative_total
        print(f'Negative accuracy (all others): {negative_accuracy:.3f} ({negative_correct}/{negative_total})')
    else:
        print(f'Negative accuracy (all others): No images found')
        negative_accuracy = 0.0
    
    print('='*50)
    
    # Save combined results
    if args.save_txt:
        with open(f"./quantitative/{args.sks_name}/acc.txt", 'a') as f:
            f.write(f'\nCOMBINED RESULTS:\n')
            f.write(f'Positive accuracy ({args.sks_name}): {positive_accuracy:.3f}\n')
            f.write(f'Negative accuracy (all others): {negative_accuracy:.3f}\n')