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
        for idx, (inline, weline) in enumerate(zip(infile, wefile), start=1):
            bleu = sacrebleu.sentence_bleu(weline, inline)
            print(bleu)
            outputs.append((idx, bleu.score, inline.strip(), weline.strip()))
        outputs.sort(key=lambda x: -x[1])
        with open(args.out, 'w') as outfile:
            print('line_no\tbleu\tref\tsys', file=outfile)
            for idx, bleu, inline, weline in outputs:
                print('{}\t{}\t{}\t{}'.format(idx, bleu, inline, weline),
                      file=outfile)


if __name__ == '__main__':
    main()
