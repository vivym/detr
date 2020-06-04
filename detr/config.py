from detectron2.config import CfgNode as CN


def add_detr_config(cfg):
    """
        Add config for DETR.
        """
    cfg.MODEL.DETR = CN()

    # position embedding: sine, learned
    cfg.MODEL.POSITION_EMBEDDING = "sine"
