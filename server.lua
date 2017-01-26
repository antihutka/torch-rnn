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
cmd:option('-multi_count', 1)
cmd:option('-benchmark', 0)
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
local input = torch.LongTensor(1, 1)
local nextline
local current_scores
local timer
if opt.benchmark > 0 then timer = torch.Timer() end
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

function get_str_simple()
  local next_char, next_idx, next_ent
  local total_ent, length = 0, opt.maxlength
  for t = 1, opt.maxlength do
    if timer then timer:reset() end
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
    if timer then print('(' .. timer:time().real .. ')') end
  end
  if next_char ~= "\n" then io.write('\n') end
  if opt.verbose > 0 then
    print(string.format('len %d ent %f', length, total_ent/length))
  end
end

function get_str_multi()
  local states, generated = {}, {}
  local starting_state = model:getState(1)
  for i = 1, opt.multi_count do
    states[i] = { state = starting_state, scores = current_scores, text = "", value = 0, next_idx = 0 , valued = 0}
  end
  while true do
    if timer then timer:reset() end
    for k,v in ipairs(states) do
      local ni, ne = model:sampleFromScores(v.scores, opt.temperature, opt.sample)
      if #v.text >= opt.maxlength then ni = model.token_to_idx['\n'] end
      v.text = v.text .. model.idx_to_token[ni]
      v.value = v.value + ne
      v.next_idx = ni
      v.valued = v.value / #v.text
      if opt.verbose > 2 then print('Current response[' .. k .. ']: ' .. v.text .. ' (' .. v.valued .. ')') end
    end
    for i = #states,1,-1 do
      if string.sub(states[i].text, -1, -1) == "\n" then
        states[i].text = string.sub(states[i].text, 1, -2)
        if opt.verbose > 1 then print('candidate (' .. states[i].valued .. '):' .. states[i].text) end
        table.insert(generated, states[i])
        table.remove(states, i)
      end
    end
    if #states == 0 then break end
    model:setBatchSize(#states)
    input:resize(#states, 1)
    for k,v in ipairs(states) do
      model:setState(k, v.state)
      input[{k,1}] = v.next_idx
    end
    local out = model:forward(input)
    for k,v in ipairs(states) do
      states[k].state = model:getState(k)
      states[k].scores = out[{{k}}]
    end
    if timer then print('Time: ' .. timer:time().real) end
  end
  table.sort(generated, function (a,b) return a.valued < b.valued end)
  if opt.verbose > 0 then
    local lasttext
    for k,v in ipairs(generated) do
      if lasttext ~= v.text then
        lasttext = v.text
        print(string.format('C %3d %1.3f %s', k, v.valued, v.text))
      end
    end
  else
    print(generated[1].text)
  end
  model:setBatchSize(1)
  model:setState(1, generated[1].state)
end

if opt.interactive == 1 then io.stdout:setvbuf('no') else io.stdout:setvbuf('line') end
if opt.multi_count > 1 then get_str = get_str_multi else get_str = get_str_simple end

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
