require 'torch'
require 'nn'

require 'LanguageModel'

torch.setdefaulttensortype('torch.FloatTensor')

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
cmd:option('-verbose', 0)
cmd:option('-color', 0)
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
local input = torch.LongTensor(1, 1)
local nextline
local current_scores
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

function put_str(s)
  local x = model:encode_string(s):view(1, -1)
  local length = x:size(2)
  local sampled = torch.LongTensor(1, length)
  sampled[{{}, {1, length}}]:copy(x)
  current_scores = model:forward(x)[{{}, {length, length}}]
end

function sq(s)
  io.write('\027[' .. s .. 'm')
end

function set_color(color)
  if opt.color > 0 then
    local code
    if color == nil then sq(0)
    elseif color < 0.1 then sq(34) -- blue
    elseif color < 0.2 then sq(1) sq(34) -- bright blue
    elseif color < 0.4 then sq(32) -- green
    elseif color < 0.8 then sq(31) -- red
    elseif color < 1.6 then sq(1) sq(33) -- bright yellow
    elseif color < 3.2 then  sq(1) -- bright
    else sq(1) sq(4) end -- bright underlined
  end
end

function get_str()
  local next_char, next_idx, next_ent
  local total_ent, length = 0, opt.maxlength
  for t = 1, opt.maxlength do
    next_idx, next_ent = model:sampleFromScores(current_scores, opt.temperature, opt.sample)
    total_ent = total_ent + next_ent
    next_char = model.idx_to_token[next_idx]
    set_color(next_ent)
    io.write(next_char)
    set_color()
    input[1] = next_idx
    current_scores = model:forward(input)
    if next_char == "\n" then
      length = t
      break
    end
  end
  if next_char ~= "\n" then io.write('\n') end
  if opt.verbose > 0 then
    print(string.format('len %d ent %f', length, total_ent/length))
  end
end

if opt.interactive == 1 then io.stdout:setvbuf('no') else io.stdout:setvbuf('line') end

put_str(opt.start_text .. "\n")

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
