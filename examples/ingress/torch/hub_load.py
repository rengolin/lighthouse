# RUN: %PYTHON %s

"""
Example demonstrating how to load a PyTorch Hub model to MLIR using Lighthouse
without initializing the model class on the user's side.

The script loads the model from the hub, then uses 'lighthouse.ingress.torch.import_from_model'.
"""

import argparse
import torch
from lighthouse.ingress.torch import import_from_model
from mlir import ir

# https://docs.pytorch.org/docs/stable/hub.html#torch.hub.load
def load_with_rand(repo: str, model: str, pretrained: bool = True):
  model = torch.hub.load(repo, model, pretrained=pretrained, trust_repo=True)
  input_shape = next(model.parameters()).shape
  sample_input = torch.randn(*input_shape)
  ir_context = ir.Context()
  return import_from_model(model, sample_args=(sample_input,), ir_context=ir_context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., 'pytorch/vision/resnet50')",
    )
    args = parser.parse_args()

    if args.model is None:
      models = ["pytorch/vision/resnet18",
                "huggingface/google-bert/bert-base-german-cased"]
    else:
      models = [args.model]

    for model_name in models:
      repo = model_name.split('/')[0] + '/' + model_name.split('/')[1]
      model = model_name.split('/')[2]
      print(f"Loading model from hub: {repo}, {model}")
      mlir_module_ir: ir.Module = load_with_rand(repo, model)
      print(f"Loaded MLIR module for model: {model_name}")
      print("\n\nModule dump:")
      mlir_module_ir.dump()

