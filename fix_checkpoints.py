
import torch

for name in ['contrastive_best', 'triplet_random_best', 'triplet_hard_best']:
    path = f'final_model.pth/{name}.pth'
    ck = torch.load(path, map_location='cpu')
    while 'model_state_dict' in ck and len(ck) == 1:
        ck = ck['model_state_dict']
    key_map = {
        'backbone.conv1': 'backbone.0',
        'backbone.bn1': 'backbone.1',
        'backbone.layer1': 'backbone.4',
        'backbone.layer2': 'backbone.5',
        'backbone.layer3': 'backbone.6',
        'backbone.layer4': 'backbone.7',
        'projection.4.weight': 'projection.weight',
        'projection.4.bias': 'projection.bias',
    }
    skip_keys = {'projection.0.weight', 'projection.0.bias', 'projection.1.weight', 'projection.1.bias'}
    new_ck = {}
    for k, v in ck.items():
        if k in skip_keys:
            continue
        new_key = k
        for old, new in key_map.items():
            if k == old or k.startswith(old + '.'):
                new_key = k.replace(old, new, 1)
                break
        new_ck[new_key] = v
    torch.save({'model_state_dict': new_ck}, path)
    print(f'Fixed {name}, projection keys:', [k for k in new_ck.keys() if 'projection' in k])