require 'torch'
require 'nn'

require 'LanguageModel'

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-verbose', 0)
cmd:option('-bytes', 1)
cmd:option('-maxlength', 512)
cmd:option('-interactive', 0)
cmd:option('-autoreply', 0)
cmd:option('-start_text', '')
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
local next_idx = 1
local input = torch.LongTensor(1, 1)
local nextline
--local output = torch.LongTensor(1, 1)

if opt.interactive == 1 then
  local ok, readline = pcall(require, 'readline')
  local history = (os.getenv('HOME') or os.getenv('USERPROFILE')) .. '/.nnbot_history'
  if ok then
    readline.setup()
    readline.read_history(history)
    nextline = function()
      local l = readline.readline("> ")
      readline.add_history(l)
      readline.write_history(history)
      return l
    end
  else
    nextline = function()
      io.write("> ")
      io.flush()
      return io.read('*line')
    end
  end
else
  nextline = function() return io.read('*line') end
end

if opt.bytes == 1 then model:convertTables() end
model:evaluate()
model:resetStates()

function idx_from_scores(scores)
  local nc
  if opt.sample == 0 then
    _, nc = scores:max(3)
    nc = nc[{{}, {}, 1}]
  else
    local probs = torch.div(scores, opt.temperature):double():exp():squeeze()
    probs:div(torch.sum(probs))
    nc = torch.multinomial(probs, 1):view(1, 1)
  end
  return nc[1][1]
end

function put_str(s)
  local x = model:encode_string(s):view(1, -1)
  local length = x:size(2)
  local sampled = torch.LongTensor(1, length)
  sampled[{{}, {1, length}}]:copy(x)
  local scores = model:forward(x)[{{}, {length, length}}]
  next_idx = idx_from_scores(scores)
end

function get_str()
  for t = 1, opt.maxlength do
    local next_char = model.idx_to_token[next_idx]
    io.write(next_char)
    input[1] = next_idx
    local out = model:forward(input)
    next_idx = idx_from_scores(out)
    if next_char == "\n" then
      break
    end
  end
  io.write('\n')
end

if opt.interactive == 1 then io.stdout:setvbuf('no') else io.stdout:setvbuf('line') end

if opt.start_text ~= '' then put_str(opt.start_text .. "\n") end

while true do
  local line = nextline()
  if line == nil then
    break
  elseif line == "" then
    if opt.interactive == 1 then io.write("< ") end
    get_str()
  else
    put_str(line .. "\n")
    if opt.autoreply == 1 then
      io.write("< ")
      get_str()
    end
  end
end
