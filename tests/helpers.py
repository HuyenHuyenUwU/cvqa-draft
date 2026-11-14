"""Cấu hình nhỏ chạy trên CPU cho test: dùng đúng lớp thật (resnet50/BERT/Qwen2) với trọng số ngẫu nhiên."""
from bone_cvqa.commons.config import deep_merge, load_config


def tiny_config(**overrides):
    config = load_config(["base", "data", "model", "training", "inference"])
    tiny = {
        "device": "cpu", "d_model": 32,
        "external_models": {
            "vision_encoder": {"provider": "tiny_random", "device": "cpu"},
            "question_encoder": {"provider": "tiny_random", "device": "cpu", "max_length": 64, "hidden_size": 48},
            "medical_llm": {"provider": "tiny_random", "device": "cpu", "hidden_size": 64},
        },
        "image_processing": {"preprocess": {"image_size": 64}},
        "cross_modal_fusion": {"n_layers": 1, "n_heads": 2, "dropout": 0.0},
        "decoder": {"max_prompt_tokens": 96, "max_answer_tokens": 12},
        "train": {"epochs": 2, "batch_size": 4, "grad_accum_steps": 2, "lr": 1e-3, "amp": False, "num_workers": 0,
                  "log_every": 1000, "max_train_batches": 3, "max_val_batches": 2, "early_stopping_patience": 0},
        "inference": {"generation": {"max_new_tokens": 6}},
        "seed": 0,
    }
    return deep_merge(deep_merge(config, tiny), overrides)
