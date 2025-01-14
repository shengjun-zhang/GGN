from dataclasses import dataclass
from typing import Literal, Optional, List
import os
import torch
import torchvision
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid, get_world_rays
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg
from src.geometry.projection import project, unproject

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding
from .common.gaussians import build_covariance
from ...misc.sh_rotation import rotate_sh


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    view_matching: bool

class GaussianGraph(nn.Module):
    def __init__(self, in_channels, out_channels, window_size=1, gamma=0.1):
        super().__init__()
        self.window_size = window_size
        self.gamma = gamma
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        
    def forward(self, means, depths, gs_feats, context):
        '''
        Input:
            means: position of gaussians (b, v, h, w, xyz)
            depths: depths of pixels (b, v, h, w, srf, dpt)
            gs_feats: features of gaussians (b, v, h, w, c)
        Output:
            gs_feats: features of gaussians (b, v, h, w, c')
        '''
        b, v, h, w, xyz = means.shape
        device = means.device
        
        # build graph
        for i in range(b):
            view_feat_list = []
            for j in range(v):
                sq_j_feat = rearrange(gs_feats[i, j], "h w c -> (h w) c", h=h, w=w)
                norm = 1
                for k in range(v):                    
                    if j != k and j >= k - self.window_size and j <= k - self.window_size:
                        sq_k_feat = rearrange(gs_feats[i, k], "h w c -> (h w) c", h=h, w=w)
                        points_ndc, valid_z = project(points=means[i,j].reshape(-1, 3),
                                                    intrinsics=context["intrinsics"][i,k],
                                                    extrinsics=context["extrinsics"][i,k],
                                                    epsilon=1e-8)
                        valid_x = (points_ndc[:, 0] >= 0) & (points_ndc[:, 0] < 1)
                        valid_y = (points_ndc[:, 1] >= 0) & (points_ndc[:, 1] < 1)
                        
                        # convert ndc to pixel
                        points2d = torch.zeros_like(points_ndc)
                        points2d[:, 0] = (points_ndc[:, 0]) * w
                        points2d[:, 1] = (points_ndc[:, 1]) * h
                        points2d = points2d.floor().long()
                        mask = valid_x & valid_y & valid_z
                        update_feat = torch.zeros(sq_k_feat.shape, dtype=sq_k_feat.dtype, device=device) # (h*w, c)
                        query_2d = torch.chunk(points2d[mask], 2, dim=-1) # ((M, 1), (M, 1))
                        query_2d = (query_2d[1], query_2d[0])
                        update_feat[mask] = gs_feats[i,k][query_2d[0], query_2d[1], :].squeeze(1) # (h*w, c)
                        sq_j_feat = sq_j_feat + update_feat * self.gamma * torch.sum(mask) / (h * w)
                        norm = norm + self.gamma * torch.sum(mask) / (h * w)

                j_feat = rearrange(sq_j_feat / norm, "(h w) c -> h w c", h=h, w=w)
                view_feat_list.append(j_feat.unsqueeze(0)) # (1, h, w, c)
                
            gs_feats[i] = torch.cat(view_feat_list, dim=0) 
        
        # network
        gs_feats = rearrange(gs_feats, "b v h w c -> (b v) c h w")
        gs_feats = self.net(gs_feats)
        gs_feats = rearrange(gs_feats, "(b v) c h w -> b v h w c", b=b, v=v)       
        
        return gs_feats


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            view_matching=cfg.view_matching
        )
        
        # Gaussians prediction: covariance, color
        gau_in = cfg.depth_unet_feat_dim + 3 + cfg.d_feature
        self.to_gaussians = nn.Sequential(
            nn.Conv1d(gau_in, cfg.num_surfaces * (self.gaussian_adapter.d_in + 2) * 2, 1),
            nn.GELU(),
            nn.Conv1d(
                cfg.num_surfaces * (self.gaussian_adapter.d_in + 2) * 2, cfg.num_surfaces * (self.gaussian_adapter.d_in), 1
            ),
        )
        
        d_sh = (self.cfg.gaussian_adapter.sh_degree + 1)**2
        self.register_buffer(
            "sh_mask",
            torch.ones((d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.gaussian_adapter.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        self.gaussian_graph = GaussianGraph(gau_in, gau_in)
        
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    
    def view_fusion(
        self,
        context,
        depths,
        opacities,
        gaussian_features,
        global_step
    ):
        # fusion weight for training
        if global_step < 1e4:
            fw = 0
        elif global_step < 5e4:
            fw = 0.1
        else:
            fw = 0.2
        gamma = 0.5
        
        b, v, _, h, w = context["image"].shape
        device = depths.device
        
        # means of gaussians
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        xy_ray = repeat(xy_ray, "r () xy -> b v r srf () xy", b=b, v=v, srf=self.cfg.num_surfaces)
        extrinsics = rearrange(context["extrinsics"], "b v i j -> b v () () () i j")
        intrinsics = rearrange(context["intrinsics"], "b v i j -> b v () () () i j")
        origins, directions = get_world_rays(xy_ray, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]
        means = rearrange(means, "b v r srf spp xyz -> b v (r srf spp) xyz",)
        means = rearrange(means, "b v (h w) xyz -> b v h w xyz", h=h, w=w)
        
        depths = rearrange(depths, "b v (h w) srf dpt -> b v h w srf dpt", h=h, w=w)
        opacities = rearrange(opacities, "b v (h w) srf dpt -> b v h w srf dpt", h=h, w=w)
        gaussian_features = rearrange(gaussian_features, "(v b) c h w -> b v h w c", v=v, b=b)

        # gaussian graph
        gaussian_features = self.gaussian_graph(means, depths, gaussian_features, context)
        
        # pooling
        gs_mean_list, gs_feat_list, gs_opacity_list, gs_offset_list, gs_depth_list = [], [], [], [], []           
        for i in range(b):
            gs_mean, gs_feat, gs_weight, gs_opacity, gs_depth = None, None, None, None, None
            gs_offset = torch.zeros((v)).long()
            for j in range(v):
                v_mean = means[i, j] # (h, w, xyz)
                v_depth = depths[i, j].squeeze(-2) # (h, w, dpt)
                v_opacity = opacities[i, j].squeeze(-2) # (h, w, dpt)
                v_feat = gaussian_features[i, j] # (h, w, c)
                
                # preserve gaussians in the first node
                if j == 0:
                    gs_mean = rearrange(v_mean, "h w xyz -> (h w) xyz")
                    gs_feat = rearrange(v_feat, "h w c -> (h w) c")
                    gs_opacity = rearrange(v_opacity, "h w dpt -> (h w) dpt")
                    gs_depth = rearrange(v_depth, "h w dpt -> (h w) dpt")
                    gs_weight = torch.ones(gs_depth.shape, dtype=v_depth.dtype, device=device)
                else:     
                    # N points input, M points are projected on view j                                
                    points_ndc, valid_z = project(points=gs_mean,
                                                    intrinsics=context["intrinsics"][i,j],
                                                    extrinsics=context["extrinsics"][i,j],
                                                    epsilon=1e-8)
                    valid_x = (points_ndc[:, 0] >= 0) & (points_ndc[:, 0] < 1)
                    valid_y = (points_ndc[:, 1] >= 0) & (points_ndc[:, 1] < 1)
                    points2d = torch.zeros_like(points_ndc)
                    points2d[:, 0] = (points_ndc[:, 0]) * w
                    points2d[:, 1] = (points_ndc[:, 1]) * h
                    points2d = points2d.floor().long()
                    mask = valid_x & valid_y & valid_z
        
                    occupied_points = gs_mean[mask]   # (M, 3)
                    wh_query = torch.chunk(points2d[mask], 2, dim=-1)
                    occupied_image_points = v_mean[wh_query[1], wh_query[0]].squeeze() # (M, 3)
                    distances = torch.norm(occupied_points - occupied_image_points, dim=1) # (M, 1)
                    threshold = 0.5
                    mask_indices = torch.where(mask)[0]
                    invalid_index = torch.where(distances > threshold)[0]
                    invalid_mask_indices = mask_indices[invalid_index]
                    mask[invalid_mask_indices] = False                
                    update_feat = torch.zeros(gs_feat.shape, dtype=gs_feat.dtype, device=device) # (N, C)
                    query_2d = torch.chunk(points2d[mask], 2, dim=-1) # ((M, 1), (M, 1))
                    query_2d = (query_2d[1], query_2d[0])
                    update_feat[mask] = v_feat[query_2d[0], query_2d[1], :].squeeze(1) * fw
                    gs_feat = gs_feat + update_feat * 0.01
                    
                    # update weight of projected points on view j
                    update_weight = torch.zeros(gs_weight.shape, dtype=gs_weight.dtype, device=device)
                    update_weight[mask] = fw ** 2
                    gs_weight = update_weight + gs_weight
                    
                    # pixel in view j which is not occupied by above points
                    image_mask = torch.ones(v_depth.shape, dtype=mask.dtype, device=device) # (h, w, dpt)
                    image_mask[query_2d[0], query_2d[1]] = 0
                    image_mask = image_mask.reshape(-1)
                    new_mean = rearrange(v_mean, "h w xyz -> (h w) xyz")             
                    new_mean = new_mean[image_mask]
                    new_feat = rearrange(v_feat, "h w c -> (h w) c")
                    new_feat = new_feat[image_mask]
                    new_opacity = rearrange(v_opacity, "h w dpt -> (h w) dpt")
                    new_opacity = new_opacity[image_mask]
                    new_depth = rearrange(v_depth, "h w dpt -> (h w) dpt")
                    new_weight = torch.ones(new_depth.shape, dtype=gs_weight.dtype, device=device)
                    new_depth = new_depth[image_mask]
                    new_weight = new_weight[image_mask]
                    
                    # concat existing gs with new gs
                    gs_mean = torch.cat((gs_mean, new_mean), dim=0)
                    gs_feat = torch.cat((gs_feat, new_feat), dim=0)
                    gs_weight = torch.cat((gs_weight, new_weight), dim=0) 
                    gs_opacity = torch.cat((gs_opacity, new_opacity), dim=0) 
                    gs_depth = torch.cat((gs_depth, new_depth), dim=0)                
                
                gs_offset[j] = gs_mean.shape[0]
                
            gs_mean_list.append(gs_mean) 
            gs_feat = gs_feat / (gs_weight**0.5 + 1e-8)
            gs_feat = self.to_gaussians(gs_feat.transpose(1, 0))
            gs_feat_list.append(gs_feat.transpose(1, 0))
            gs_opacity_list.append(gs_opacity)
            gs_offset_list.append(gs_offset)
            gs_depth_list.append(gs_depth)   
        
        return gs_mean_list, gs_feat_list, gs_opacity_list, gs_offset_list, gs_depth_list
    
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):       
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel
        depths, densities, raw_gaussians_in = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
        )
        
        # predict opacity from densities
        opacities = self.map_pdf_to_opacity(densities, global_step) / gpp   
        
        # multi-view fusion
        mean_list, raw_gaussians_list, opacity_list, offset_list, depths_list \
            = self.view_fusion(context, depths, opacities, raw_gaussians_in, global_step)
         
        gs_list = []
        
        for i in range(b):
            mean = mean_list[i]
            raw_gaussian = raw_gaussians_list[i]
            opacity = opacity_list[i].squeeze(-1)
            offset = offset_list[i]
            depth = depths_list[i]
            d_sh = (self.cfg.gaussian_adapter.sh_degree + 1)**2
            scales, rotations, sh = raw_gaussian.split((3, 4, 3 * d_sh), dim=-1)

            # Map scale features to valid scale range.
            scale_max = self.cfg.gaussian_adapter.gaussian_scale_max
            scale_min = self.cfg.gaussian_adapter.gaussian_scale_min
            scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            intrinsic = context["intrinsics"][i, 0]
            scales = scales * depth * (intrinsic[0,0]/w + intrinsic[1,1]/h)

            # Normalize the quaternion features to yield a valid quaternion.
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)

            # Apply sigmoid to get valid colors.
            sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3) * self.sh_mask # (n, xyz, d_sh)

            # Build covariance
            covariances = build_covariance(scales, rotations) # (n, 3, 3)

            # Compute harmonics
            harmonics = []
            for j in range(v):
                c2w_rot = context["extrinsics"][i, j, :3, :3].unsqueeze(0)
                if j == 0:
                    covariances[:offset[j]] = c2w_rot @ covariances[:offset[j]] @ c2w_rot.transpose(-1, -2)
                    harmonics.append(rotate_sh(sh[:offset[j]], c2w_rot))
                else:
                    if offset[j] - offset[j-1] != 0:
                        covariances[offset[j-1]:offset[j]] = c2w_rot @ covariances[offset[j-1]:offset[j]] @ c2w_rot.transpose(-1, -2)
                        harmonics.append(rotate_sh(sh[offset[j-1]:offset[j]], c2w_rot))
            harmonics = torch.cat(harmonics, dim=0)
           
            gs_list.append((mean, covariances, harmonics, opacity, scales, rotations))
            
        return gs_list
        
    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
