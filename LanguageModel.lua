require 'torch'
require 'nn'
require 'TemporalAdapter'
require 'VanillaRNN'
require 'LSTM'
require 'GRU'
require 'GRIDGRU'

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

  self.rnns = {}
  self.net = nn.Sequential()

  self.net:add(nn.LookupTable(V, D))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    elseif self.model_type == 'gru' then
      rnn = nn.GRU(prev_dim, H)
    elseif self.model_type == 'gridgru' then
      rnn = nn.GRIDGRU(D, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      self.net:add(nn.TemporalAdapter(nn.BatchNormalization((self.model_type == 'gridgru') and D or H)))
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout, nil, true))
    end
  end

  if self.model_type == 'gridgru' then
    self.net:add(nn.TemporalAdapter(nn.Linear(D, V)))
  else
    self.net:add(nn.TemporalAdapter(nn.Linear(H, V)))
  end
  self.has_temporal_adapter = true
end


function LM:updateOutput(input)
  if not self.has_temporal_adapter then self:patch_ta() end
  return self.net:forward(input)
end


function LM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  self.net:training()
  parent.training(self)
end


function LM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end


function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


function LM:encode_string(s)
  local encoded = torch.LongTensor(#s)
  for i = 1, #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    if idx == nil then idx = self.token_to_idx[" "] end
    assert(idx ~= nil, 'Got invalid idx')
    encoded[i] = idx
  end
  return encoded
end


function LM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    if token ~= nil then s = s .. token end
  end
  return s
end


function LM:patch_ta()
  local between_views, mods = false, self.net.modules
  local i,v
  for i,v in ipairs(mods) do
    if torch.type(mods[i]) == 'nn.View' then
      between_views = not between_views
      mods[i] = nn.Identity()
    elseif between_views then
      mods[i] = nn.TemporalAdapter(mods[i])
    end
  end
  self.has_temporal_adapter = true
end

--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs.
Inputs:
- init: String of length T0
- max_length: Number of characters to sample
Returns:
- sampled: (1, max_length) array of integers, where the first part is init.
--]]
function LM:sample(kwargs, charout)
  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 1)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)

  local sampled = torch.LongTensor(1, T)
  self:resetStates()

  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    sampled[{{}, {1, T0}}]:copy(x)
    scores = self:forward(x)[{{}, {T0, T0}}]
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end
  
  local _, next_char = nil, nil
  for t = first_t, T do
    if sample == 0 then
      _, next_char = scores:max(3)
      next_char = next_char[{{}, {}, 1}]
    else
       local probs = torch.div(scores, temperature):double():exp():squeeze()
       probs:div(torch.sum(probs))
       next_char = torch.multinomial(probs, 1):view(1, 1)
    end
    sampled[{{}, {t, t}}]:copy(next_char)
    charout(self.idx_to_token[next_char[1][1]])
    scores = self:forward(next_char)
    if kwargs.stop_on_newline == 1 and self.idx_to_token[next_char[1][1]] == "\n" then
    	break
    end
  end

  self:resetStates()
  return self:decode_string(sampled[1])
end


function LM:clearState()
  self.net:clearState()
end

function LM:convertTables()
  for k,v in pairs(self.idx_to_token) do
    if (v:sub(1,1) == "[") and (v:len() == 5) then
      local newv = string.char(tonumber(v:sub(2,4)))
      self.idx_to_token[k] = newv
      self.token_to_idx[newv] = k
    end
  end
end

function LM:swappable(layers)
  for k,v in ipairs(self.rnns) do
    if layers < 2 or k <= layers then v:swappable(self.rnns[k-1]) end
  end
end

function LM:remove_grad()
  local f
  f = function(module)
    module.gradWeight = nil
    module.gradBias = nil
    if (module.net) then module.net:apply(f) end
  end
  self.net:apply(f)
end
