import os
import json
from PIL import Image

from model.utils.opt import get_default_parser
from model.utils.utils import generate_first_image
from model.text2room_pipeline import Text2RoomPipeline

import torch
from pytorch_lightning import seed_everything

@torch.no_grad()
def main(args):
    seed_everything(args.seed)

    # load trajectories
    trajectories = json.load(open(args.trajectory_file, "r"))

    # check if there is a custom prompt in the first trajectory
    # would use it to generate start image, if we have to
    if "prompt" in trajectories[0]:
        args.prompt = trajectories[0]["prompt"]

    # get first image from text prompt or saved image folder
    if args.input_image_path and os.path.isfile(args.input_image_path):
        first_image_pil = Image.open(args.input_image_path)
    else:
        first_image_pil = generate_first_image(args)
    
    first_image_pil.show()

    # load pipeline
    pipeline = Text2RoomPipeline(args, first_image_pil=first_image_pil)

    # generate using all trajectories
    offset = 1  # have the start image already
    for t in trajectories:
        pipeline.set_trajectory(t)
        offset = pipeline.generate_images(offset=offset)
    
    # save outputs before completion
    pipeline.clean_mesh()
    pipeline.save_mesh("after_generation.ply")
    
    pipeline.save_seen_trajectory_renderings(apply_noise=False, add_to_nerf_images=True)
    pipeline.save_nerf_transforms()
    pipeline.save_animations()

    print("Finished. Outputs stored in:", args.out_path)


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args([
                "--trajectory_file", "/scratch/students/2023-spring-mt-mhnowak/text2room/experimenting_trajectory.json",
                "--quick_run",
                "--skip_classical_inpainting",
                "--depth_estimator_model", "midas",
                "--depth_estimator_model_path", "/scratch/students/2023-spring-mt-mhnowak/BoostingMonocularDepth/midas/model.pt",
                "--pix2pix_model_path", "/scratch/students/2023-spring-mt-mhnowak/BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth",
                "--input_image_path", "/scratch/students/2023-spring-mt-mhnowak/text2room/street-photo-3.png",
                "--skip_depth_boosting",
                "--perform_depth_fine_tuning",
                "--number_midas_fine_tuning_epochs", "150"
            ])
    main(args)
