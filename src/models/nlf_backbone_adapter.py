import torch


class NLFBackboneAdapter:
    """
    Thin wrapper to use NLF's crop_model.backbone as the image feature encoder.

    Usage:
        adapter = NLFBackboneAdapter(nlf_model)
        feats = adapter(image)  # returns backbone features
    """

    def __init__(self, nlf_model):
        if not hasattr(nlf_model, "crop_model") or not hasattr(nlf_model.crop_model, "backbone"):
            raise AttributeError("nlf_model must expose crop_model.backbone")
        self.nlf_model = nlf_model

    def __call__(self, image: torch.Tensor, use_half: bool = True) -> torch.Tensor:
        x = image.half() if use_half else image
        return self.nlf_model.crop_model.backbone(x)

