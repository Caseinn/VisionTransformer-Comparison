import torch
import timm


def create_model(config, device, freeze_head=True):
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
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)


def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    model_size_mb = get_model_size_mb(model)

    print("=== MODEL PARAMETER ===")
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    print(f"Non-Trainable Parameters: {non_trainable_params:,}")
    print(f"Model Size (FP32):        {model_size_mb:.2f} MB")
