import ForSEplus
from ForSEplus import my_forse_class

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dirs', type=str, required=True)
parser.add_argument('--patch_file', type=str, required=True)

args = parser.parse_args()
print(args.output_dirs)
print(args.patch_file)

test_forse = my_forse_class.forse_my(args.output_dirs)
test_forse.train(epochs=200001, patches_file=args.patch_file, batch_size=16, save_interval=500)