import os
import json
from PIL import Image

from model.utils.opt import get_default_parser
from model.utils.utils import generate_first_image, save_image
from model.text2room_pipeline import Text2RoomPipeline
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline

import torch
from pytorch_lightning import seed_everything

@torch.no_grad()
def main(args):
    number_images = 1
    seed_everything(args.seed)

    model_path = os.path.join(args.models_path, "stable-diffusion-2-1")
    if not os.path.exists(model_path):
        model_path = "stabilityai/stable-diffusion-2-1"
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, generator=generator)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    pipe.set_progress_bar_config(**{
        "leave": False,
        "desc": "Generating Start Image"
    })

    output = pipe(args.prompt)
    for index, image in enumerate(output.images):
        save_image(image, f"generated_image_{args.prompt}_seed_{args.seed}", index, "generated_images")


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args([
                # "--prompt", "realistic photograph of a busy street in a city during daytime",
                # "--prompt", "",
                # "--prompt", "photograph of an outdoor scenery, night, mountains, trees, clear sky",
                # "--prompt", "POV, cave, pools, water, dark cavern, inside a cave, beautiful scenery, best quality, indoor scene",
                "--prompt", "inside a castle made of ice, beautiful photo, masterpiece",
                "--quick_run",
                "--skip_classical_inpainting",
                "--depth_estimator_model", "midas",
                "--depth_estimator_model_path", "/scratch/students/2023-spring-mt-mhnowak/BoostingMonocularDepth/midas/model.pt",
                "--pix2pix_model_path", "/scratch/students/2023-spring-mt-mhnowak/BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth",
                # "--input_image_path", "/scratch/students/2023-spring-mt-mhnowak/text2room/street-photo-3.png",
                "--skip_depth_boosting",
                # "--perform_depth_fine_tuning",
                # "--number_midas_fine_tuning_epochs", "150"
                "--seed", "55",
                "--min_triangles_connected", "1000",
                "--edge_threshold", "0.01",
                "--surface_normal_threshold", "0.01",
            ])
    main(args)
