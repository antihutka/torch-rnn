require 'torch'
require 'nn'

require 'LanguageModel'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-length', 2000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
cmd:option('-hide_start_text', 0)
cmd:option('-read_start_text', 0)
cmd:option('-stop_on_newline', 0)
cmd:option('-bytes', 0)
local opt = cmd:parse(arg)

if opt.read_start_text == 1 then
	opt.start_text = io.read("*all")
end

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

model:remove_grad()

local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end
if opt.verbose == 1 then print(msg) end
if opt.bytes == 1 then model:convertTables() end

model:evaluate()

if opt.hide_start_text == 0 then
	io.write(opt.start_text)
end

io.stdout:setvbuf('no') 

local sample = model:sample(opt, io.write)
print('')
