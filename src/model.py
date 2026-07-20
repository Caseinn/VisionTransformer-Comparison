import torch
import timm


def create_model(config, device, freeze_head=True):
    """Create and prepare a pretrained timm model for CIFAR-10.

    The model is instantiated with ten output classes, moved to the requested
    device, and optionally frozen so only the classification head remains
    trainable. A single dummy forward pass is printed to confirm the output
    shape before training starts.

    Args:
        config: Model configuration containing the timm model id and display name.
        device: Torch device where the model and dummy input should be placed.
        freeze_head: If ``True``, freeze all parameters except those whose names
            include ``"head"``.

    Returns:
        The initialized PyTorch model on ``device``.
    """
    print(f"Loading {config.name} from timm...")
    model = timm.create_model(config.timm_model, pretrained=True, num_classes=10)

    if freeze_head:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    model = model.to(device)
    print(
        "Model output shape test:",
        model(torch.randn(1, 3, 224, 224).to(device)).shape,
    )
    return model


def get_model_size_mb(model):
    """Estimate a model's in-memory parameter and buffer size.

    Args:
        model: PyTorch module whose parameters and buffers should be measured.

    Returns:
        Total parameter and buffer storage in mebibytes, assuming the tensors'
        current dtypes.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)


def print_model_params(model):
    """Print a summary of model parameter counts and approximate size.

    Args:
        model: PyTorch module to summarize.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    model_size_mb = get_model_size_mb(model)

    print("=== MODEL PARAMETER ===")
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    print(f"Non-Trainable Parameters: {non_trainable_params:,}")
    print(f"Model Size (FP32):        {model_size_mb:.2f} MB")
