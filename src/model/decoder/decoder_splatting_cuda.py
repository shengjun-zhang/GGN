from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gs_list,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        colors = []
        for i in range(b):
            gs = gs_list[i]
            means, covariances, harmonics, opacities, scales, rotations = gs
            import time
            time_start = time.time()
            color = render_cuda(
                extrinsics[i],
                intrinsics[i],
                near[i],
                far[i],
                image_shape,
                repeat(self.background_color, "c -> v c", v=v),
                repeat(means[:131072, :], "g xyz -> v g xyz", v=v),
                repeat(covariances[:131072, :, :], "g i j -> v g i j", v=v),
                repeat(harmonics[:131072, :, :], "g c d_sh -> v g c d_sh", v=v),
                repeat(opacities[:131072], "g -> v g", v=v),
            )
            time_end = time.time()
            render_time = time_end - time_start
            
            # import json
            # try:
            #     with open('render_data.json', 'r') as file:
            #         existing_data = json.load(file)
            # except FileNotFoundError:
            #     existing_data = []
            # existing_data.append(render_time)

            # with open('render_data.json', 'w') as file:
            #     json.dump(existing_data, file)
                
            colors.append(color)
        if b==1:
            colors = color
        else:
            colors = torch.cat(colors, dim=0)
        # color = render_cuda(
        #     rearrange(extrinsics, "b v i j -> (b v) i j"),
        #     rearrange(intrinsics, "b v i j -> (b v) i j"),
        #     rearrange(near, "b v -> (b v)"),
        #     rearrange(far, "b v -> (b v)"),
        #     image_shape,
        #     repeat(self.background_color, "c -> (b v) c", b=b, v=v),
        #     repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
        #     repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
        #     repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
        #     repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        # )
        colors = rearrange(colors, "(b v) c h w -> b v c h w", b=b, v=v)

        return DecoderOutput(
            colors,
            None
            if depth_mode is None
            else self.render_depth(
                gs_list, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
        )

    def render_depth(
        self,
        gs_list,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        results = []
        for i in range(b):
            gs = gs_list[i]
            means, covariances, harmonics, opacities, scales, rotations = gs
            result = render_depth_cuda(
                extrinsics[i],
                intrinsics[i],
                near[i],
                far[i],
                image_shape,
                repeat(means, "g xyz -> v g xyz", v=v),
                repeat(covariances, "g i j -> v g i j", v=v),
                repeat(opacities, "g -> v g", v=v),
                mode=mode
            )
            results.append(result)
        if b==1:
            results = result
        else:    
            results = torch.cat(result, dim=0)
        # result = render_depth_cuda(
        #     rearrange(extrinsics, "b v i j -> (b v) i j"),
        #     rearrange(intrinsics, "b v i j -> (b v) i j"),
        #     rearrange(near, "b v -> (b v)"),
        #     rearrange(far, "b v -> (b v)"),
        #     image_shape,
        #     repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
        #     repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
        #     repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        #     mode=mode,
        # )
        return rearrange(results, "(b v) h w -> b v h w", b=b, v=v)
