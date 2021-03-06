# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import h5py
import codecs


parser = argparse.ArgumentParser()
parser.add_argument('--input_txt', default='data/tiny-shakespeare.txt')
parser.add_argument('--input_json', default='data/index.json')
parser.add_argument('--output_h5', default='data/tiny-shakespeare.h5')
parser.add_argument('--output_json', default='data/index.json')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--freeze-vocab', action='store_true')
parser.add_argument('--encoding', default='utf-8')
args = parser.parse_args()

def load_index(fname):
  if os.path.exists(fname):
    with open(fname, "r") as infile:
      return json.load(infile)
  else:
    return None

def save_index(fname, token_to_idx):
  # Dump a JSON file for the vocab
  json_data = {
    'token_to_idx': token_to_idx,
    'idx_to_token': {v: k for k, v in token_to_idx.iteritems()},
  }
  with open(fname, 'w') as f:
    json.dump(json_data, f)

if __name__ == '__main__':
  if args.encoding == 'bytes': args.encoding = None

  # First go the file once to see how big it is and to build the vocab
  token_to_idx = {}
  total_size = 0
  # Load token index mapping if exists
  index = load_index(args.input_json)
  if index is not None:
    token_to_idx = index['token_to_idx']

  # If we're in bytes mode, convert [numbers] back to characters
  if args.encoding is None:
    new_token_to_idx = {}
    for key in token_to_idx:
      if len(key) > 1:
        num = int(key[1:4])
        new_token_to_idx[chr(num)] = token_to_idx[key]
      else:
        new_token_to_idx[key] = token_to_idx[key]
    token_to_idx = new_token_to_idx

  with codecs.open(args.input_txt, 'r', args.encoding) as f:
    for line in f:
      total_size += len(line)
      for char in line:
        if char not in token_to_idx:
          if args.freeze_vocab:
            raise Exception('Tried to expand frozen vocabulary: "' + line + '" / ' + str(ord(char)))
          token_to_idx[char] = len(token_to_idx) + 1

  # Now we can figure out the split sizes
  val_size = int(args.val_frac * total_size)
  test_size = int(args.test_frac * total_size)
  train_size = total_size - val_size - test_size
 
  if not args.quiet:
    print 'Total vocabulary size: %d' % len(token_to_idx)
    print 'Total tokens in file: %d' % total_size
    print '  Training size: %d' % train_size
    print '  Val size: %d' % val_size
    print '  Test size: %d' % test_size

  # Choose the datatype based on the vocabulary size
  dtype = np.uint8
  if len(token_to_idx) > 255:
    dtype = np.uint32
  if not args.quiet:
    print 'Using dtype ', dtype

  # Just load data into memory ... we'll have to do something more clever
  # for huge datasets but this should be fine for now
  train = np.zeros(train_size, dtype=dtype)
  val = np.zeros(val_size, dtype=dtype)
  test = np.zeros(test_size, dtype=dtype)
  splits = [train, val, test]

  # Go through the file again and write data to numpy arrays
  split_idx, cur_idx = 0, 0
  with codecs.open(args.input_txt, 'r', args.encoding) as f:
    for line in f:
      for char in line:
        splits[split_idx][cur_idx] = token_to_idx[char]
        cur_idx += 1
        if cur_idx == splits[split_idx].size:
          split_idx += 1
          cur_idx = 0

  # Write data to HDF5 file
  with h5py.File(args.output_h5, 'w') as f:
    f.create_dataset('train', data=train)
    f.create_dataset('val', data=val)
    f.create_dataset('test', data=test)

  # For 'bytes' encoding, replace non-ascii characters so the json dump
  # doesn't crash
  if args.encoding is None:
    new_token_to_idx = {}
    for token, idx in token_to_idx.iteritems():
      if ord(token) > 127:
        new_token_to_idx['[%d]' % ord(token)] = idx
      else:
        new_token_to_idx[token] = idx
    token_to_idx = new_token_to_idx

  # Dump a JSON file for the vocab
  save_index(args.output_json, token_to_idx)
