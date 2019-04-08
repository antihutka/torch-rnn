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
cmd:option('-relevance_sampling', 0)
cmd:option('-relevance_selection', 0)
cmd:option('-gpu', 0)
cmd:option('-commands', 0)
cmd:option('-savedir', 'savestate')
cmd:option('-ksm', 0)
cmd:option('-lineprefix', '')
cmd:option('-chop', 32)
cmd:option('-print_newline_prob', 0)
cmd:option('-soft_newline_start', 400)
cmd:option('-soft_newline_mult', 0.03)
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
local input = torch.LongTensor(1, 1)
local nextline
local current_scores
local timer
if opt.benchmark > 0 then timer = torch.Timer() end
--local output = torch.LongTensor(1, 1)
local use_relevance = false
if opt.relevance_sampling > 0 or opt.relevance_selection > 0 then use_relevance = true end

if opt.ksm > 0 then
  local ksm = require 'util.ksm'
  ksm.make_parameters_mergeable(model)
end

if checkpoint.is_mapped then
  model:unmapTensors(opt.checkpoint)
end

if opt.gpu > 0 then
  require 'cutorch'
  require 'cunn'
  model:cuda()
end

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
  if opt.chop > 0 and opt.chop < length then
    for i = 1,length,opt.chop do
      local j = math.min(length, i + opt.chop - 1)
      if opt.verbose > 0 then print('chop', i, j) end
      current_scores = model:forward(x[{{}, {i, j}}]) [{{}, {-1, -1}}]
    end
  else
    current_scores = model:forward(x)[{{}, {length, length}}]
  end
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
  local next_char, next_idx, next_ent, next_prob, all_probs
  local total_ent, length = 0, opt.maxlength
  for t = 1, opt.maxlength do
    local newline_boost = 0
    if timer then timer:reset() end
    if opt.soft_newline_start < t then
      newline_boost = (t - opt.soft_newline_start) * opt.soft_newline_mult
      current_scores[{1, 1, model.token_to_idx['\n']}] = current_scores[{1, 1, model.token_to_idx['\n']}] + newline_boost
    end
    next_idx, next_ent, next_prob, all_probs = model:sampleFromScores(current_scores, opt.temperature, opt.sample)
    if opt.print_newline_prob > 0 then
      local newline_prob = all_probs[model.token_to_idx['\n']]
      if newline_prob > 0.01 then
        io.write(string.format('\027[34m[%3.2f %d +%.1f]\027[0m', newline_prob, t, newline_boost))
      end
    end
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
    states[i] = { state = starting_state, scores = current_scores, text = "", value = 0, next_idx = 0 , valued = 0, valuer = 0, valuerd = 0}
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
      if v.scores_r then
        local rp = model:probsFromScores(v.scores_r, opt.temperature)
        v.valuer = v.valuer - math.log(rp[ni])
        v.valuerd = v.valuer / #v.text
      end
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
    local batchsize = #states
    if use_relevance then batchsize = 2 * batchsize end
    model:setBatchSize(batchsize)
    input:resize(batchsize, 1)
    for k,v in ipairs(states) do
      model:setState(k, v.state)
      input[{k,1}] = v.next_idx
      if use_relevance then input[{k+#states,1}] = v.next_idx end
      if v.state_r then model:setState(k+#states, v.state_r) end
    end
    local out = model:forward(input)
    for k,v in ipairs(states) do
      states[k].state = model:getState(k)
      states[k].scores = out[{{k}}]
      if use_relevance then
        states[k].state_r = model:getState(k+#states)
        states[k].scores_r = out[{{k+#states}}]
      end
      if opt.relevance_sampling > 0 then states[k].scores:add(-opt.relevance_sampling, out[{{k+#states}}]) end
    end
    if timer then print('Time: ' .. timer:time().real) end
  end
  for k,v in ipairs(generated) do
    v.valuef = v.valued - opt.relevance_selection * v.valuerd
  end
  table.sort(generated, function (a,b) return a.valuef < b.valuef end)
  if opt.verbose > 0 then
    print('')
    local lasttext
    for k,v in ipairs(generated) do
      if lasttext ~= v.text then
        lasttext = v.text
        print(string.format('C %3d %1.3f %1.3f %1.3f %s', k, v.valuef, v.valued, v.valuerd, v.text))
      end
    end
  else
    print(generated[1].text)
  end
  model:setBatchSize(1)
  model:setState(1, generated[1].state)
  put_str("\n")
end

if opt.interactive == 1 then io.stdout:setvbuf('no') else io.stdout:setvbuf('line') end
if opt.multi_count > 1 then get_str = get_str_multi else get_str = get_str_simple end

put_str(opt.start_text .. "\n")

local initial_state = model:getState(1)
local initial_scores = current_scores:clone()

local function checkname(n)
  if (not n:find('^[A-Za-z0-9_-]+$')) then
    error('bad file name')
  end
end

local function loadstate(fn)
  local s = torch.load(opt.savedir .. '/' .. fn .. '.state')
  model:setState(1, s.state)
  current_scores = s.scores
end

local function runcmd(l)
  local cmd, arg = l:match(" *([^ ]+) +([^ ].*)")
  if cmd == 'save' then
    checkname(arg)
    torch.save(opt.savedir .. '/' .. arg .. '.state', {state = model:getState(1), scores = current_scores})
  elseif cmd == 'load' then
    checkname(arg)
    if not pcall(function () loadstate(arg) end) then
      io.stderr:write('error loading state ' .. arg .. '\n')
      model:setState(1, initial_state)
      current_scores = initial_scores:clone()
    end
  elseif cmd == 'reset' or l == 'reset' then
    model:setState(1, initial_state)
    current_scores = initial_scores:clone()
  else
    error('bad command')
  end
end

while true do
  local line = nextline()
  if line == nil then
    break
  elseif line == "" then
    if opt.interactive == 1 then io.write("< ") end
    if opt.lineprefix ~= '' then put_str(opt.lineprefix) io.write(opt.lineprefix) end
    get_str()
  elseif line:sub(1,2) == "/!" and opt.commands > 0 then
    runcmd(line:sub(3))
  else
    put_str(line .. "\n")
    if opt.autoreply == 1 then
      io.write("< ")
      if opt.lineprefix ~= '' then put_str(opt.lineprefix) io.write(opt.lineprefix) end
      get_str()
    end
  end
end
