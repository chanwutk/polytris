import torch


RELEVANT_CATEGORIES = ['car', 'bus', 'truck']


def collapse_classifier(full_weights, last_layer: torch.nn.Linear) -> torch.nn.Linear:
    # categories = full_weights.meta["categories"]
    # relevant_categories = [i for i, c in enumerate(categories) if c in RELEVANT_CATEGORIES]
    # irrelevant_categories = [i for i in range(len(categories)) if i not in relevant_categories]

    # rel_weight = last_layer.weight.data[relevant_categories].sum(dim=1, keepdim=True)
    # irr_weight = last_layer.weight.data[irrelevant_categories].sum(dim=1, keepdim=True)

    new_layer = torch.nn.Linear(last_layer.in_features, 1,
                                bias=last_layer.bias is not None,
                                device=last_layer.weight.device,
                                dtype=last_layer.weight.dtype)
    # new_layer.weight = rel_weight - irr_weight
    # if last_layer.bias is not None:
    #     rel_bias = last_layer.bias.data[relevant_categories].sum()
    #     irr_bias = last_layer.bias.data[irrelevant_categories].sum()
    #     new_layer.bias.data[:] = rel_bias - irr_bias
    return new_layer