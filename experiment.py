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
    # pipeline.save_animations()
    pipeline.save_videos()

    print("Finished. Outputs stored in:", args.out_path)


# street_1 = [
#     "--prompt", "photograph of an inviting street in a city during daytime, realistic",
#     "--seed", "43",
#     # "--disparity_min", "0.25",
#     "--disparity_max", "600.0",
# ]

# street_2 = [ # Usable <- choice
#     "--prompt", "photograph of an inviting street in a city during daytime, realistic",
#     "--seed", "52",
#     # "--disparity_min", "0.25",
#     "--disparity_max", "600.0",
# ]

# street_2_1 = [ # Usable <- choice
#     "--prompt", "photograph of an inviting street in a city during daytime, realistic",
#     "--seed", "52",
#     "--disparity_min", "0.5",
#     "--disparity_max", "2",
# ]

street_1 = [ # Usable <- choice
    "--prompt", "photograph of an inviting street in a city during daytime, realistic",
    "--seed", "52",
    "--prompt_name", "street_1",
]

# street_3 = [ # Usable <- choice
#     "--prompt", "inviting street in a big city during the evening, photograph, realistic",
#     "--seed", "56",
#     # "--disparity_min", "0.25",
#     "--disparity_max", "600.0",
# ]

street_2 = [ # Usable <- choice
    "--prompt", "inviting street in a big city during the evening, photograph, realistic",
    "--seed", "56",
    "--prompt_name", "street_2",
]

# street_4 = [
#     "--prompt", "POV, inviting urban park in a small city with cars and people during daytime, photograph, realistic",
#     "--seed", "55",
#     # "--disparity_min", "0.25",
#     "--disparity_max", "600.0",
# ]

street_3 = [ # Usable <- choice
    "--prompt", "inviting busy cental station in a large city, photograph, realistic",
    "--seed", "55",
    "--prompt_name", "street_3",
]

# scenery_2 = [ ## Usable
#     "--prompt", "photograph of an outdoor scenery, daytime, mountains, trees, clear sky",
#     "--seed", "43",
#     "--disparity_min", "0.005",
#     "--disparity_max", "600.0",
#     # "--edge_threshold", "0.00", # "0.01"
#     # "--surface_normal_threshold", "0.00", # "0.01"
# ]

# scenery_3 = [ ## Usable <- choice
#     "--prompt", "beautiful view of a forest, daylight, high quality",
#     "--seed", "44",
#     "--disparity_min", "0.005",
#     "--disparity_max", "600.0",
#     # "--edge_threshold", "0.00", # "0.01"
#     # "--surface_normal_threshold", "0.00", # "0.01"
# ]


# scenery_3_1 = [
#     "--prompt", "beautiful view of a forest, daylight, high quality",
#     "--seed", "44",
#     "--disparity_min", "0.005",
#     "--disparity_max", "600.0",
#     # "--edge_threshold", "0.00", # "0.01"
#     # "--surface_normal_threshold", "0.00", # "0.01"
# ]

# scenery_4 = [ ## Usable
#     "--prompt", "photograph an outdoor scenery, mountain, lake, trees, daytime, clear sky",
#     "--seed", "42",
#     "--disparity_min", "0.005",
#     "--disparity_max", "600.0",
#     # "--edge_threshold", "0.00", # "0.01"
#     # "--surface_normal_threshold", "0.00", # "0.01"
# ]

# scenery_5 = [ ## Usable <- choice
#     "--prompt", "photograph of an outdoor scenery, evening, mountains, trees, clear sky",
#     "--seed", "44",
#     "--disparity_min", "0.005",
#     "--disparity_max", "600.0",
#     # "--edge_threshold", "0.00", # "0.01"
#     # "--surface_normal_threshold", "0.00", # "0.01"
# ]

scenery_1 = [ ## Usable <- choice
    "--prompt", "outdoor scenery, mountain, lake, trees, daytime, clear sky",
    "--seed", "42",
    "--prompt_name", "scenery_1"
]

scenery_2 = [ ## Usable <- choice
    "--prompt", "beautiful view of a forest, daylight, high quality",
    "--seed", "44",
    "--prompt_name", "scenery_2"
]

scenery_3 = [ ## Usable <- choice
    "--prompt", "photograph of an outdoor scenery, evening, mountains, trees, clear sky",
    "--seed", "44",
    "--prompt_name", "scenery_3"
]

scene_scape_1 = [ ## Usable <- choice
    "--prompt", "POV, cave, pools, water, dark cavern, inside a cave, beautiful scenery, best quality, indoor scene",
    "--seed", "53",
    "--prompt_name", "scene_scape_1"
]

scene_scape_2 = [ ## Usable <- choice
    "--prompt", "walkthrough, inside a medieval forge, metal, fire, beautiful photo, masterpiece, indoor scene",
    "--seed", "42",
    "--prompt_name", "scene_scape_2"
]

scene_scape_3 = [ # Usable <- choice
    "--prompt", "inside a castle made of ice, beautiful photo, masterpiece",
    "--seed", "54",
    "--prompt_name", "scene_scape_3"
]

if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args([
                "--skip_classical_inpainting",
                "--quick_run",
                "--depth_estimator_model", "midas",
                "--depth_estimator_model_path", "/scratch/students/2023-spring-mt-mhnowak/BoostingMonocularDepth/midas/model.pt",
                "--pix2pix_model_path", "/scratch/students/2023-spring-mt-mhnowak/BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth",
                "--skip_depth_boosting",
                "--use_midas_v3_from_hub",
                "--out_path", "final_output_2",
                # "--trajectory_file", "/scratch/students/2023-spring-mt-mhnowak/text2room/experiment_trajectories/urban_2.json",
                "--trajectory_file", "/scratch/students/2023-spring-mt-mhnowak/text2room/final_experimenting_trajectory_3.json",
                # "--prompt", "photograph of an inviting street in a city during daytime, realistic",
                "--negative_prompt", "text, writings, signs, text, white border, photograph border, artifacts, blur, smooth texture, foggy, fog, bad quality, distortions, unrealistic, distorted image, watermark, signature, fisheye look, windows, people, crowd, view, chandelier",
                "--disparity_min", "1",
                "--disparity_max", "4",
                # "--skip_depth_alignment",
                # "--input_image_path", "/scratch/students/2023-spring-mt-mhnowak/text2room/street-photo-3.png",
                # "--perform_depth_fine_tuning",
                # "--number_midas_fine_tuning_epochs", "150"
                # "--min_triangles_connected", "15000",
                # "--edge_threshold", "0.00", # "0.01"
                # "--surface_normal_threshold", "0.00", # "0.01"
            ] + street_3)
    main(args)
