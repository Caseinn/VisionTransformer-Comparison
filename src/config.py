from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration values for one CIFAR-10 vision transformer experiment.

    Attributes:
        name: Human-readable model name used in reports and output folders.
        timm_model: Model identifier passed to ``timm.create_model``.
        pretrained_source: Dataset or checkpoint family used for pretrained weights.
        checkpoint_name: File name used when saving the best fine-tuned weights.
        patch_token_start: Index where image patch tokens begin in the feature tensor.
            Distilled models usually reserve earlier tokens for class/distillation tokens.
        has_attention_maps: Whether the model supports the attention-style patch
            importance visualization used by this project.
    """

    name: str
    timm_model: str
    pretrained_source: str
    checkpoint_name: str
    patch_token_start: int
    has_attention_maps: bool

    @property
    def output_dir(self):
        """Return the results directory for this model.

        Returns:
            Relative path under ``results/`` where checkpoints, metrics, and
            figures for this model are stored.
        """
        return f"results/{self.name}"


MODEL_CONFIGS = {
    'deit_tiny': ModelConfig(
        name="DeiT Tiny",
        timm_model='deit_tiny_distilled_patch16_224.fb_in1k',
        pretrained_source='ImageNet-1k',
        checkpoint_name='deit_tiny_cifar10_best.pth',
        patch_token_start=2,
        has_attention_maps=True,
    ),
    'eva02_tiny': ModelConfig(
        name="EVA-02 Tiny",
        timm_model='eva02_tiny_patch14_224.mim_in22k',
        pretrained_source='ImageNet-22k',
        checkpoint_name='eva02_tiny_cifar10_best.pth',
        patch_token_start=1,
        has_attention_maps=True,
    ),
    'pit_tiny': ModelConfig(
        name="PiT Tiny",
        timm_model='pit_ti_distilled_224.in1k',
        pretrained_source='ImageNet-1k',
        checkpoint_name='pit_tiny_cifar10_best.pth',
        patch_token_start=2,
        has_attention_maps=True,
    ),
}
