from collections import OrderedDict
import torch

def get_parameters(model):
    """
    Converts the model's weights into the flat list-of-NumPy-arrays format
    Flower's NumPyClient protocol expects. state_dict() preserves insertion
    order, so this order is consistent and reversible with set_parameters.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    """
    Reverse of get_parameters: takes a list of NumPy arrays (e.g. after
    Flower's server aggregates weights from multiple hospital nodes) and
    loads them back into the model's state_dict.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

if __name__ == "__main__":
    from unet_model import build_model

    model = build_model()
    params = get_parameters(model)
    print(f"Number of parameter arrays: {len(params)}")
    print(f"First array shape: {params[0].shape}, dtype: {params[0].dtype}")

    # Round-trip test: set the same parameters back, confirm no errors
    set_parameters(model, params)
    print("Round-trip set_parameters succeeded, no shape/key mismatches.")