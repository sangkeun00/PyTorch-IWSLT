import argparse

import sacrebleu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref')
    parser.add_argument('sys')
    parser.add_argument('out')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.ref) as infile, open(args.sys) as wefile:
        outputs = []
        for inline, weline in zip(infile, wefile):
            bleu = sacrebleu.sentence_bleu(weline, inline)
            print(bleu)
            outputs.append((bleu.score, inline.strip(), weline.strip()))
        outputs.sort(key=lambda x: -x[0])
        with open(args.out, 'w') as outfile:
            for bleu, inline, weline in outputs:
                print('{}\t{}\t{}'.format(bleu, inline, weline), file=outfile)


if __name__ == '__main__':
    main()
