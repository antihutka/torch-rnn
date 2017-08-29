require 'torch'
require 'nn'

require 'LanguageModel'
require 'util.DataLoader'

torch.setdefaulttensortype('torch.FloatTensor')

local utils = require 'util.utils'


local cmd = torch.CmdLine()

cmd:option('-checkpoint', '')
cmd:option('-split', 'val')
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-input_h5', '')
cmd:option('-seq_length', 0)
cmd:option('-batch_size', 0)
local opt = cmd:parse(arg)


-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  print(string.format('Running with OpenCL on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  print 'Running in CPU mode'
end

-- Load the checkpoint and model
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:type(dtype)
local crit = nn.CrossEntropyCriterion():type(dtype)

if opt.input_h5 ~= '' then checkpoint.opt.input_h5 = opt.input_h5 end
if opt.seq_length > 0 then checkpoint.opt.seq_length = opt.seq_length end
if opt.batch_size > 0 then checkpoint.opt.batch_size = opt.batch_size end
if checkpoint.opt.seq_offset == nil then checkpoint.opt.seq_offset = 0 end

-- Load the vocab and data
local loader = DataLoader(checkpoint.opt)
local N, T = checkpoint.opt.batch_size, checkpoint.opt.seq_length

-- Evaluate the model on the specified split
model:evaluate()
model:resetStates()
local num = loader.split_sizes[opt.split]
local loss = 0
local lossstring = ''
for i = 1, num do
  print(string.format('%s batch %d / %d %s', opt.split, i, num, lossstring))
  local x, y = loader:nextBatch(opt.split)
  x = x:type(dtype)
  y = y:type(dtype):view(N * T)
  local scores = model:forward(x):view(N * T, -1)
  loss = loss + crit:forward(scores, y)
  lossstring = string.format('average loss = %f', loss / i)
end
loss = loss / num
print(string.format('%s loss = %f', opt.split, loss))
