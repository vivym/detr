import torch
import torch.nn as nn

from detectron2.structures import ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone

from .position_encoding import build_position_encoding


@META_ARCH_REGISTRY.register()
class DETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.mask_on = cfg.MODEL.MASK_ON

        self.backbone = build_backbone(cfg)
        self.position_embedding = build_position_encoding(cfg)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)

        masks = {
            name: None # TODO: mask on
            for name, x in features.items()
        }

        pos = {}
        for name, x in features.items():
            pos[name] = self.position_embedding(x, masks[name]).to(x)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
