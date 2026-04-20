"""
RelTR model definition — inference only.

Original: https://github.com/yrcong/RelTR/blob/main/models/reltr.py
License: Apache 2.0

Stripped training-only code (SetCriterion, matcher) for cleaner inference.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .backbone import build_backbone
from .transformer import build_transformer


class RelTR(nn.Module):
    """RelTR: Relation Transformer for Scene Graph Generation."""

    def __init__(self, backbone, transformer, num_classes, num_rel_classes,
                 num_entities, num_triplets, aux_loss=False):
        super().__init__()
        self.num_entities = num_entities
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.entity_embed = nn.Embedding(num_entities, hidden_dim * 2)
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim * 3)
        self.so_embed = nn.Embedding(2, hidden_dim)

        # entity prediction
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # mask head
        self.so_mask_conv = nn.Sequential(
            torch.nn.Upsample(size=(28, 28)),
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128))

        # predicate classification
        self.rel_class_embed = MLP(hidden_dim * 2 + 128, hidden_dim, num_rel_classes + 1, 2)

        # subject/object classification and box regression
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, samples):
        from .misc import NestedTensor, nested_tensor_from_tensor_list

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, hs_t, so_masks, _ = self.transformer(
            self.input_proj(src), mask,
            self.entity_embed.weight, self.triplet_embed.weight,
            pos[-1], self.so_embed.weight)

        so_masks = so_masks.detach()
        so_masks = self.so_mask_conv(
            so_masks.view(-1, 2, src.shape[-2], src.shape[-1])
        ).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2], -1)
        so_masks = self.so_mask_fc(so_masks)

        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)

        outputs_class = self.entity_class_embed(hs)
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()

        outputs_class_sub = self.sub_class_embed(hs_sub)
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()

        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid()

        outputs_class_rel = self.rel_class_embed(
            torch.cat((hs_sub, hs_obj, so_masks), dim=-1))

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "sub_logits": outputs_class_sub[-1],
            "sub_boxes": outputs_coord_sub[-1],
            "obj_logits": outputs_class_obj[-1],
            "obj_boxes": outputs_coord_obj[-1],
            "rel_logits": outputs_class_rel[-1],
        }
        return out


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    """Build RelTR model for inference."""
    num_classes = 151 if args.dataset != "oi" else 289
    num_rel_classes = 51 if args.dataset != "oi" else 31

    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = RelTR(
        backbone, transformer,
        num_classes=num_classes,
        num_rel_classes=num_rel_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss)

    model.to(device)
    return model
