require 'torch'
require 'nn'

local layer, parent = torch.class('nn.History', 'nn.Module')

function layer:__init(history_depth)
  parent.__init(self)
  self.history_depth = history_depth
  self.gradInput = torch.Tensor()
  self.prevInputs = torch.Tensor()
end

function layer:updateOutput(input)
  local N, T, ID = input:size(1), input:size(2), input:size(3)
  local HD = self.history_depth

  local output, prev = self.output, self.prevInputs

  if prev:nElement() == 0 then
    prev:resize(N, HD, ID):zero()
  else
    assert(self.prevInputs:size(1) == N, 'batch size changed')
  end

  output:resize(N, T, ID*(HD+1))
  prev:resize(N, HD, ID)

  for i = 0, HD do
    if i < T then
      output[{{}, {1+i, T}, {i*ID+1, (i+1)*ID}}]:copy(input[{{}, {1, T-i}}])
    end
    if i > 0 then
      local e = 0
      if T < i then e = i - T end
      output[{{}, {1, i-e}, {i*ID+1, (i+1)*ID}}]:copy(prev[{{}, {-i, HD-e}}])
    end
  end
  if HD > 0 then
    if HD <= T then
      prev:copy(input[{{}, {-HD, T}}])
    else
      for i = 1, HD-T do
        prev[{{}, {i,i}}]:copy(prev[{{}, {i+1, i+1}}])
      end
      prev[{{}, {-T, -1}}]:copy(input)
    end
  end

  return output
end

function layer:updateGradInput(input, gradOutput)
  local N, T, ID = input:size(1), input:size(2), input:size(3)
  local HD = self.history_depth

  local gradInput = self.gradInput
  gradInput:resizeAs(input)

  gradInput:copy(gradOutput[{{}, {}, {1, ID}}])
  for i = 1, HD do
    if i < T then
      gradInput[{{}, {1, T-i}}]:add(gradOutput[{{}, {1+i, T}, {i*ID+1, (i+1)*ID}}])
    end
  end

  return gradOutput
end

function layer:clearState()
  self.gradInput:set()
  self.prevInputs:set()
  self.output:set()
end
