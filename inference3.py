import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from PIL import Image, ImageFilter
from datetime import datetime

from model.pipeline import CatVTONPix2PixPipeline
from inference2 import repaint, repaint2, to_pil_image, InferenceDataset, VITONHDTestDataset, DressCodeTestDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="timbrooks/instruct-pix2pix",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON-MaskFree",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The datasets to use for evaluation.",
    )
    parser.add_argument(
        "--data_root_path", 
        type=str, 
        required=True,
        help="Path to the dataset to evaluate."
    )
    parser.add_argument(
        "--paired_data_path", 
        type=str, 
        required=True,
        help="Path to the paired data."
    )    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=555, help="A seed for reproducible evaluation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="The batch size for evaluation."
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to perform.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="The scale of classifier-free guidance for inference.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--eval_pair",
        action="store_true",
        help="Whether or not to evaluate the pair.",
    )
    parser.add_argument(
        "--concat_eval_results",
        action="store_true",
        help="Whether or not to  concatenate the all conditions into one image.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--concat_axis",
        type=str,
        choices=["x", "y", 'random'],
        default="y",
        help="The axis to concat the cloth feature, select from ['x', 'y', 'random'].",
    )
    parser.add_argument(
        "--enable_condition_noise",
        action="store_true",
        default=True,
        help="Whether or not to enable condition noise.",
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


@torch.no_grad()
def main():
    args = parse_args()
    # Pipeline
    pipeline = CatVTONPix2PixPipeline(
        attn_ckpt_version=args.dataset_name,
        attn_ckpt=args.resume_path,
        base_ckpt=args.base_model_path,
        weight_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        use_tf32=args.allow_tf32,
        device='cuda',
        skip_safety_check=True,
    )    
    # Dataset
    if args.dataset_name == "vitonhd":
        dataset = VITONHDTestDataset(args)
    elif args.dataset_name == "dresscode":
        dataset = DressCodeTestDataset(args)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}.")
    print(f"Dataset {args.dataset_name} loaded, total {len(dataset)} pairs.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )
    # Inference
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    data_pair_type = "paired" if args.eval_pair else "unpaired"
    args.output_dir = os.path.join(
        args.output_dir, 
        f"{args.dataset_name}-{args.height}-{data_pair_type}-{timestamp}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for batch in tqdm(dataloader):
        person_images = batch['person']
        cloth_images = batch['cloth']
        masks = batch['mask']
        results = pipeline(
            image=person_images,
            condition_image=cloth_images,
            height=args.height,
            width=args.width,            
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        )        
        
        if args.concat_eval_results or args.repaint:
            person_images = to_pil_image(person_images)
            cloth_images = to_pil_image(cloth_images)
            masks = to_pil_image(masks)
        for i, result in enumerate(results):
            # person_name = batch['person_name'][i]
            output_name = batch['output_name'][i]
            output_path = os.path.join(args.output_dir, output_name)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))           
            if args.repaint:
                person_path, mask_path = dataset.data[batch['index'][i]]['person'], dataset.data[batch['index'][i]]['mask']
                person_image= Image.open(person_path).resize(result.size, Image.LANCZOS)
                mask = Image.open(mask_path).resize(result.size, Image.NEAREST)
                ori_result = result.copy()
                result = repaint2(person_image, mask, result)
            if args.concat_eval_results:
                concat_output_path = os.path.join(args.output_dir, "concat-{}".format(output_name))
                if not os.path.exists(os.path.dirname(concat_output_path)):
                    os.makedirs(os.path.dirname(concat_output_path))
                w, h = result.size
                concated_result = Image.new('RGB', (w*4, h))
                concated_result.paste(person_images[i], (0, 0))
                concated_result.paste(cloth_images[i], (w, 0))  
                concated_result.paste(result, (w*2, 0))
                if ori_result is not None:
                    concated_result.paste(ori_result, (w*3, 0))                
                concated_result.save(concat_output_path)
            result.save(output_path)

if __name__ == "__main__":
    main()