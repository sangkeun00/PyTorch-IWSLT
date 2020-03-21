import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--keys', nargs='+')
    return parser.parse_args()


def main():
    args = parse_args()
    d = torch.load(args.path)
    if args.keys:
        for key in args.keys:
            d = d[key]

    for key, value in sorted(d.items()):
        print(key + '\t' + ', '.join([str(v) for v in value.size()]))


if __name__ == '__main__':
    main()
