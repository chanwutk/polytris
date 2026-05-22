import torch


# Kept for imports / docs; ImageNet names use phrases like "school bus", not the literal word "car".
RELEVANT_CATEGORIES = ['car', 'bus', 'truck']

# Substrings in torchvision ``meta["categories"]`` strings (lowercased) for road-vehicle classes.
# Leading spaces / phrases avoid false positives (e.g. "car" inside "carpet").
_VEHICLE_FRAGMENTS = (
    ' bus', ' truck', ' jeep', ' minivan', ' limousine', ' racer',
    ' sports car', ' convertible', ' ambulance', ' fire engine',
    ' minibus', ' trolleybus', ' pickup', ' tow truck', ' trailer truck',
    ' moving van', ' police van', ' golfcart', ' golf cart',
    ' station wagon', ' beach wagon', ' taxicab', ' taxi', ' cab',
    ' freight car', ' passenger car', ' go-kart', ' snowplow', ' mobile home',
)


def collapse_classifier(full_weights, last_layer: torch.nn.Linear) -> torch.nn.Linear:
    categories = full_weights.meta.get('categories')
    if categories is None:
        raise ValueError('full_weights.meta must contain "categories"')
    if not isinstance(last_layer, torch.nn.Linear):
        raise TypeError('collapse_classifier expects last_layer to be nn.Linear')

    n = len(categories)
    if last_layer.out_features != n:
        raise ValueError(f'Linear out_features {last_layer.out_features} != len(categories) {n}')

    relevant = [
        i for i, c in enumerate(categories)
        if any(t in c.lower() for t in _VEHICLE_FRAGMENTS)
    ]
    if not relevant:
        raise ValueError('No vehicle-like ImageNet classes matched; extend _VEHICLE_FRAGMENTS if needed.')

    irrelevant = [i for i in range(n) if i not in set(relevant)]

    w = last_layer.weight.data
    # Each row is one class; sum rows then subtract (was wrongly using sum(dim=1) on row slices).
    rel_w = w[relevant].sum(dim=0)
    irr_w = w[irrelevant].sum(dim=0)
    new_w = (rel_w - irr_w).unsqueeze(0)

    new_layer = torch.nn.Linear(
        last_layer.in_features,
        1,
        bias=last_layer.bias is not None,
        device=last_layer.weight.device,
        dtype=last_layer.weight.dtype,
    )
    new_layer.weight.data.copy_(new_w)
    if last_layer.bias is not None:
        b = last_layer.bias.data
        new_layer.bias.data.copy_(b[relevant].sum() - b[irrelevant].sum())

    return new_layer
