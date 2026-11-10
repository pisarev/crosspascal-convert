# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Export the Gemma SentencePiece BPE vocabulary for Pascal.

Split normalized text into individual characters before ranked BPE merges.
Replace a character missing from the vocabulary with its UTF-8 byte tokens.
Binary files store little-endian offsets, token IDs and code points.
"""
import argparse
import json
import os
import struct
import sys


def meaning_bytes(tok):
    """Return the bytes represented by this token."""
    if len(tok) == 6 and tok.startswith('<0x') and tok.endswith('>'):
        return bytes([int(tok[3:5], 16)])
    return tok.encode('utf-8')


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args()


def main():
    args = parse_arguments()
    with open(args.tokenizer, encoding='utf-8') as f:
        spec = json.load(f)

    model = spec['model']
    if model['type'] != 'BPE':
        sys.exit('Expected a BPE tokenizer, got %s' % model['type'])
    if not model.get('byte_fallback'):
        sys.exit('Expected the Gemma tokenizer to use byte fallback')

    vocab = model['vocab']
    merges = model['merges']

    top = max(vocab.values())
    words = [None] * (top + 1)
    for word, number in vocab.items():
        words[number] = word
    holes = [i for i, w in enumerate(words) if w is None]
    if holes:
        sys.exit('Missing vocabulary IDs: %s' % holes[:8])

    raw = [meaning_bytes(w) for w in words]


    chars = []
    for word, number in vocab.items():
        if len(word) == 1:
            chars.append((ord(word), number))
    chars.sort()


    byte_id = [-1] * 256
    for word, number in vocab.items():
        if len(word) == 6 and word.startswith('<0x') and word.endswith('>'):
            byte_id[int(word[3:5], 16)] = number


    triples = []
    unknown = 0
    for item in merges:
        if isinstance(item, str):
            left, _, right = item.partition(' ')
        else:
            left, right = item[0], item[1]
        joined = left + right
        if left in vocab and right in vocab and joined in vocab:
            triples.append((vocab[left], vocab[right], vocab[joined]))
        else:
            unknown += 1

    out_dir = args.out
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    offsets = []
    at = 0
    with open(os.path.join(out_dir, 'tokenizer-tokens.bin'), 'wb') as f:
        for piece in raw:
            f.write(piece)
            offsets.append((at, len(piece)))
            at += len(piece)
    with open(os.path.join(out_dir, 'tokenizer-tokens.index'), 'wb') as f:
        for begin, length in offsets:
            f.write(struct.pack('<II', begin, length))

    with open(os.path.join(out_dir, 'tokenizer-merges.bin'), 'wb') as f:
        for l, r, j in triples:
            f.write(struct.pack('<iii', l, r, j))

    with open(os.path.join(out_dir, 'tokenizer-chars.bin'), 'wb') as f:
        for cp, i in chars:
            f.write(struct.pack('<Ii', cp, i))

    with open(os.path.join(out_dir, 'tokenizer-bytes.bin'), 'wb') as f:
        for i in byte_id:
            f.write(struct.pack('<i', i))


    specials = spec.get('added_tokens', [])
    with open(os.path.join(out_dir, 'tokenizer-specials.txt'), 'w',
              encoding='utf-8', newline='\n') as f:
        f.write('%d\n' % len(specials))
        for item in specials:
            one = item['content'].replace('\n', '\\n')
            f.write('%d %d %s\n' % (item['id'], 1 if item.get('special') else 0, one))


    bos = -1
    pp = spec.get('post_processor') or {}
    for name, info in (pp.get('special_tokens') or {}).items():
        if name == '<bos>' and info.get('ids'):
            bos = info['ids'][0]
    norm = (spec.get('normalizer') or {})
    with open(os.path.join(out_dir, 'tokenizer.index'), 'w',
              encoding='utf-8', newline='\n') as f:
        f.write('style sentencepiece\n')
        f.write('tokens %d\n' % len(raw))
        f.write('merges %d\n' % len(triples))
        f.write('chars %d\n' % len(chars))
        f.write('specials %d\n' % len(specials))
        f.write('bytes %d\n' % at)
        f.write('bos %d\n' % bos)
        f.write('space_meta 2581\n')            
        f.write('normalizer %s\n' % norm.get('type', 'none'))

    missing = [b for b in range(256) if byte_id[b] < 0]
    print(f'Vocabulary tokens: {len(raw)}, decoded bytes: {at}')
    print(f'Single-character tokens: {len(chars)}')
    print(f'Byte fallback: {256 - len(missing)} of 256, missing: {missing}')
    print(f'Merges: {len(triples)}, skipped: {unknown}')
    print(f'Added tokens: {len(specials)}, BOS ID: {bos}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
