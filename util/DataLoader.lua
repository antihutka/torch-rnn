require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(kwargs)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  self.seq_offset = utils.get_kwarg(kwargs, 'seq_offset')
  self.shuffle = utils.get_kwarg(kwargs, 'shuffle_data')
  local N, T = self.batch_size, self.seq_length

  -- Just slurp all the data into memory
  local splits = {}
  local f = hdf5.open(h5_file, 'r')
  splits.train = f:read('/train'):all()
  splits.val = f:read('/val'):all()
  splits.test = f:read('/test'):all()

  local hb = math.floor(self.batch_size/2)
  splits.train_off = splits.train[{{1+hb, -1}}]
  splits.train = splits.train[{{1, -1-hb}}]

  self.x_splits = {}
  self.y_splits = {}
  self.split_sizes = {}
  for split, v in pairs(splits) do
    local num = v:nElement()
    local extra = num % (N * T)

    -- Ensure that `vy` is non-empty
    if extra == 0 then
      extra = N * T
    end

    -- Chop out the extra bits at the end to make it evenly divide
    local vx = v[{{1, num - extra}}]:view(N, -1, T):transpose(1, 2):clone()
    local vy = v[{{2, num - extra + 1}}]:view(N, -1, T):transpose(1, 2):clone()

    self.x_splits[split] = vx
    self.y_splits[split] = vy
    self.split_sizes[split] = vx:size(1)
  end

  self.split_idxs = {train=1, val=1, test=1}
  self.shuffle_order = torch.LongTensor()
end


function DataLoader:nextBatch(split)
  local idx = self.split_idxs[split]
  assert(idx, 'invalid split ' .. split)
  local x = self.x_splits[split][idx]
  local y = self.y_splits[split][idx]
  local z
  if self.shuffle > 0 and split == 'train' then
    local so = self.shuffle_order
    if idx == 1 then
      so:randperm(self.split_sizes[split])
    end
    x = self.x_splits[split][so[idx]]
    y = self.y_splits[split][so[idx]]

    if so[idx] > 1 then
      z = self.x_splits[split][so[idx]-1]
    end
    if idx > 1 and so[idx] == so[idx-1]+1 then
      z = nil
    end
  end
  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
    if split == 'train' and self.seq_offset > 0 then
      local tx, ty = self.x_splits.train, self.y_splits.train
      self.x_splits.train, self.y_splits.train = self.x_splits.train_off, self.y_splits.train_off
      self.x_splits.train_off, self.y_splits.train_off = tx, ty
    end
  else
    self.split_idxs[split] = idx + 1
  end
  return x, y, z
end

