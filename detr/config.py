from detectron2.config import CfgNode as CN


def add_detr_config(cfg):
    """
        Add config for DETR.
        """
    cfg.MODEL.DETR = CN()

    # position embedding: sine, learned
    cfg.MODEL.POSITION_EMBEDDING = "sine"

    # ---------------------------------------------------------------------------- #
    # Transformer
    # ---------------------------------------------------------------------------- #

    # Number of encoding layers in the transformer
    cfg.MODEL.NUM_ENCODER_LAYERS = 6
    # Number of decoding layers in the transformer
    cfg.MODEL.NUM_DECODER_LAYERS = 6
    # Intermediate size of the feedforward layers in the transformer blocks
    cfg.MODEL.DIM_FEEDFORWARD = 2048
    # Size of the embeddings (dimension of the transformer)
    cfg.MODEL.DIM_HIDDEN = 256
    # Dropout applied in the transformer
    cfg.MODEL.DROPOUT = 0.1
    # Number of attention heads inside the transformer's attentions
    cfg.MODEL.NUM_HEADS = 8
    # Number of query slots
    cfg.MODEL.NUM_QUERIES = 100
    # pre norm
    cfg.PRE_NORM_ON = False
