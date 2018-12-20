import os
from glob import glob

import torch

path = './result'
file_paths = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.pt'))]

results = []
for file_path in file_paths:
    if 'result' in file_path:
        name = file_path.split('/')[-1].split('.')[0]
        best_accuracy = torch.load(file_path)
        results.append([name, best_accuracy])

print(results)
