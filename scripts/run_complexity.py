import sys
sys.path.insert(0, 'src')
from evaluation.complexity import profile_variant

r = profile_variant('hybrid')
for k, v in r.items():
    print(f'{k}: {v}')
