# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import subprocess
from typing import Optional
import torch
import numpy as np
from cog import BasePredictor, Input, Path, BaseModel
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import read_video_frames, vis_sequence_depth, save_video


MODEL_CACHE = "model_cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/Tencent/DepthCrafter/{MODEL_CACHE}.tar"
)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


class ModelOutput(BaseModel):
    npz: Optional[Path]
    depth_video: Path


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            f"{MODEL_CACHE}/tencent/DepthCrafter",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        self.pipe = DepthCrafterPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        # self.pipe = DepthCrafterPipeline.from_pretrained(
        #     f"{MODEL_CACHE}/stabilityai/stable-video-diffusion-img2vid-xt",
        #     # unet=unet,
        #     torch_dtype=torch.float16,
        #     variant="fp16",
        # )
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

    def predict(
        self,
        video: Path = Input(description="Input video"),
        num_denoising_steps: int = Input(
            description="Number of denoising steps", ge=1, le=25, default=10
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=1.2, default=1.2
        ),
        max_res: int = Input(
            description="Maximum resolution",
            default=1024,
            choices=[
                512,
                576,
                640,
                704,
                768,
                832,
                896,
                960,
                1024,
                1088,
                1152,
                1216,
                1280,
                1344,
                1408,
                1472,
                1536,
                1600,
                1664,
                1728,
                1792,
                1856,
                1920,
                1984,
                2048,
            ],
        ),
        process_length: int = Input(
            description="Number of frames to process", ge=1, le=280, default=60
        ),
        target_fps: int = Input(description="fps of the output video", default=15),
        window_size: int = Input(description="Window size", default=110),
        overlap: int = Input(description="Overlap size", default=15),
        save_npz: bool = Input(description="Save npz file", default=False),
        datast: str = Input(
            description="Assigned resolution for specific dataset evaluation",
            default="open",
            choices=["open", "sintel", "scannet", "kitti", "bonn", "nyu"],
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        frames, target_fps = read_video_frames(
            str(video), process_length, target_fps, max_res, datast
        )
        print(f"==> frames shape: {frames.shape}")

        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=True,
            ).frames[0]

        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        vis = vis_sequence_depth(res)

        if save_npz:
            npz_path = "/tmp/out.npz"
            np.savez_compressed(npz_path, depth=res)
        depth_path = "/tmp/out.mp4"
        save_video(vis, depth_path, fps=target_fps)
        save_video(vis, "out.mp4", fps=target_fps)

        return ModelOutput(
            npz=Path(npz_path) if save_npz else None,
            depth_video=Path(depth_path),
        )
