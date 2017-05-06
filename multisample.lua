require 'torch'
require 'nn'

require 'LanguageModel'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-bytes', 1)
cmd:option('-start_text', '\n')
cmd:option('-length', 1024)
cmd:option('-count', 16)
cmd:option('-print_every', 10)
cmd:option('-output_file', 'outputs/output-#.txt')
cmd:option('-verbose', 0)
cmd:option('-forcelayer', 0)
cmd:option('-forcevalue', 1)
local opt = cmd:parse(arg)

local timer = torch.Timer()
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
local input = torch.LongTensor(opt.count, 1)
local outtext = {}
local outputs
local newline_idx = model.token_to_idx['\n']
local outfiles = {}

for i = 1, opt.count do
  outtext[i] = ""
  outfiles[i] = io.open(string.gsub(opt.output_file, "#", string.format("%03d", i)), "w")
  outfiles[i]:setvbuf('no')
end

if opt.bytes == 1 then model:convertTables() end

local inp = model:encode_string(opt.start_text):view(1, -1)
outputs = model:forward(inp)[{{}, {-1, -1}}]:expand(opt.count, 1, model.vocab_size)

local state = model:getState(1)
model:setBatchSize(opt.count)
for i = 1, opt.count do
  model:setState(i, state)
end

print(string.format('Initialization complete in %.2fs', timer:time().real))
timer:reset()

for i = 1, opt.length do
  for j = 1, opt.count do
    local ni = model:sampleFromScores(outputs[{{j}}], opt.temperature, 1)
    if ni == newline_idx then
      if opt.verbose > 0 then print(string.format("%3d:%s", j, outtext[j])) end
      outfiles[j]:write(outtext[j] .. '\n')
      outtext[j] = ""
    else
      outtext[j] = outtext[j] .. model.idx_to_token[ni]
    end
    input[{j,1}] = ni
  end

  if opt.forcelayer > 0 then
    for i = 1, opt.count do
      local l = model.rnns[opt.forcelayer]:getState(i)
      l[i] = opt.forcevalue
    end
  end

  outputs = model:forward(input)
  if i % opt.print_every == 0 then
    local t, b = timer:time().real, opt.count * opt.print_every
    print(string.format("%8d/%8d %.2f %.1f/s", i, opt.length, t, b/t))
    timer:reset()
  end
end

for i = 1, opt.count do
  outfiles[i]:write(outtext[i])
  outfiles[i]:close()
end
