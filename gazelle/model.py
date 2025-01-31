import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math

import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone
import torchvision.transforms.functional as F

class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim,
                num_heads=8,
                mlp_ratio=4,
                drop_path=0.1)
                for i in range(num_layers)
                ])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.head_token = nn.Embedding(1, self.dim)
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            self.inout_token = nn.Embedding(1, self.dim)

    def forward(self, input):
        # input["images"]: [B, 3, H, W] tensor of images
        # input["bboxes"]: list of lists of bbox tuples [[(xmin, ymin, xmax, ymax)]] per image in normalized image coords

        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        x = self.backbone.forward(input["images"])
        x = self.linear(x)
        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img) # repeat image features along people dimension per image
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device) # [sum(N_p), 32, 32]
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :] # slice off inout tokens from scene tokens

        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img) # resplit per image

        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        # bboxes: [[(xmin, ymin, xmax, ymax)]] - list of list of head bboxes per image
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None: # no bbox provided, use empty head map
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}

    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1 if not k.startswith("backbone")])
            keys2 = set([k for k in keys2 if not k.startswith("backbone")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)


class GazeLLE_ONNX(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim,
                num_heads=8,
                mlp_ratio=4,
                drop_path=0.1)
                for i in range(num_layers)
                ])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.head_token = nn.Embedding(1, self.dim)
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            self.inout_token = nn.Embedding(1, self.dim)

    def forward(self, image_bgr, bboxes):
        # input["images"]: [B, 3, H, W] tensor of images
        # input["bboxes"]: list of lists of bbox tuples [[(xmin, ymin, xmax, ymax)]] per image in normalized image coords

        # return transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485,0.456,0.406],
        #         std=[0.229,0.224,0.225]
        #     ),
        #     transforms.Resize(in_size),
        # ])

        mean = torch.tensor([0.485,0.456,0.406], dtype=torch.float32).reshape([1,3,1,1])
        std = torch.tensor([0.229,0.224,0.225], dtype=torch.float32).reshape([1,3,1,1])

        image_rgb = torch.cat([image_bgr[:, 2:3, ...], image_bgr[:, 1:2, ...], image_bgr[:, 0:1, ...]], dim=1)
        image_rgb = F.resize(img=image_rgb, size=(448, 448))
        image_rgb = image_rgb * 0.003921569
        image_rgb = (image_rgb - mean) / std

        num_ppl_per_img = bboxes.shape[1]

        x = self.backbone.forward(image_rgb)
        x = self.linear(x)
        x = x + self.pos_embed
        x = x * torch.ones([num_ppl_per_img,1,1,1], dtype=torch.float32)

        head_maps = self.get_input_head_maps(bboxes).to(x.device).permute(1,0,2,3) # [sum(N_p), 1, 32, 32]
        head_map_embeddings = head_maps * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            x = x[:, 1:, :] # slice off inout tokens from scene tokens

        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = x

        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes: torch.Tensor):
        # bboxes: [[(xmin, ymin, xmax, ymax)]] - list of list of head bboxes per image
        N, M, _ = bboxes.shape

        # 1. 画像サイズに基づいて bboxes のスケール変換
        bboxes_scaled = bboxes
        bboxes_scaled[..., 0:1] = bboxes_scaled[..., 0:1] * self.featmap_w  # xmin, xmax を featmap_w でスケール
        bboxes_scaled[..., 2:3] = bboxes_scaled[..., 2:3] * self.featmap_w  # xmin, xmax を featmap_w でスケール
        bboxes_scaled[..., 1:2] = bboxes_scaled[..., 1:2] * self.featmap_h  # ymin, ymax を featmap_h でスケール
        bboxes_scaled[..., 3:4] = bboxes_scaled[..., 3:4] * self.featmap_h  # ymin, ymax を featmap_h でスケール

        # 2. 整数に変換 (torch.round して int32 に変換)
        bboxes_scaled = torch.round(bboxes_scaled).to(torch.int32)  # [N, M, 4]
        xmin, ymin, xmax, ymax = bboxes_scaled.split(1, dim=2)  # [N, M] に分解

        # 3. y, x のインデックスを作成
        y_range = torch.arange(self.featmap_h, device=bboxes.device).view(1, 1, -1, 1)  # [1, 1, H, 1]
        x_range = torch.arange(self.featmap_w, device=bboxes.device).view(1, 1, 1, -1)  # [1, 1, 1, W]

        # 4. バイナリマスクの作成
        ymin = ymin[..., None]  # [N, M, 1, 1]
        ymax = ymax[..., None]  # [N, M, 1, 1]
        xmin = xmin[..., None]  # [N, M, 1, 1]
        xmax = xmax[..., None]  # [N, M, 1, 1]

        y_mask = (y_range >= ymin) & (y_range < ymax)  # [N, M, H, 1]
        x_mask = (x_range >= xmin) & (x_range < xmax)  # [N, M, 1, W]
        head_maps = y_mask & x_mask  # [N, M, H, W]

        return head_maps

    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}

    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1 if not k.startswith("backbone")])
            keys2 = set([k for k in keys2 if not k.startswith("backbone")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)

# From https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


# models
def get_gazelle_model(model_name, onnx_export=False):
    factory = {
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14(onnx_export),
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14(onnx_export),
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout(onnx_export),
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout(onnx_export),
    }
    assert model_name in factory.keys(), "invalid model name"
    if not onnx_export:
        return factory[model_name]()
    else:
        return factory[model_name]

def gazelle_dinov2_vitb14(onnx_export):
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    if not onnx_export:
        model = GazeLLE(backbone)
    else:
        model = GazeLLE_ONNX(backbone)
    return model, transform

def gazelle_dinov2_vitl14(onnx_export):
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    if not onnx_export:
        model = GazeLLE(backbone)
    else:
        model = GazeLLE_ONNX(backbone)
    return model, transform

def gazelle_dinov2_vitb14_inout(onnx_export):
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    if not onnx_export:
        model = GazeLLE(backbone)
    else:
        model = GazeLLE_ONNX(backbone, inout=True)
    return model, transform

def gazelle_dinov2_vitl14_inout(onnx_export):
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    if not onnx_export:
        model = GazeLLE(backbone)
    else:
        model = GazeLLE_ONNX(backbone, inout=True)
    return model, transform
