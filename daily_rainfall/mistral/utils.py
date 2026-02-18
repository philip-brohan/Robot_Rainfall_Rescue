# Ministral-specific utilities

from transformers import (
    FineGrainedFP8Config,
)
from peft import PeftModel

model_kwargs = dict(
    device_map="auto",  # Let torch decide how to load the model
    quantization_config=FineGrainedFP8Config(
        dequantize=True
    ),  # FP8 needs H100s. This switches to Bfloat16 - for A100s
)


# Load a saved trained model state
def load_model_from_save(
    model_dir, base_model_id, model_class, processor_class, device
):
    # Try to load a full model from model_dir first (merged/full save)
    try:
        proc = processor_class.from_pretrained(model_dir)
    except Exception:
        proc = processor_class.from_pretrained(base_model_id)

    try:
        model = model_class.from_pretrained(model_dir, **model_kwargs)
        model.to(device)
        return model, proc
    except Exception:
        # If model_dir only contains PEFT adapters, load base model then apply adapter
        model = model_class.from_pretrained(base_model_id, **model_kwargs)
        try:
            peft_model = PeftModel.from_pretrained(model, model_dir)
            peft_model.to(device)
            return peft_model, proc
        except Exception as e:
            raise RuntimeError(f"Failed to load model or PEFT from {model_dir}: {e}")
