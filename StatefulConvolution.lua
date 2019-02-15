local SC, parent = torch.class('nn.StatefulConvolution', 'nn.Module')

-- N - batches, T - time, I - input dim, O - output dim, D - dilat (not implemented)
-- (N, T, I) -> (N, T, O)

function SC:__init(I, O, K, D)
  parent.__init(self)
  self.input_dim = I
  self.output_dim = O
  self.kernel_size = K
  self.dilat = D
  self.weight = torch.Tensor(K, I, O)
  self.gradWeight = torch.Tensor(K, I, O):zero()
  self.bias = torch.Tensor(O)
  self.gradBias = torch.Tensor(O):zero()
  self.gradBiasBuffer = torch.Tensor()
  self:reset()
  
  self.input_prev = torch.Tensor()
  self.input_prev_back = torch.Tensor()
  self.gradInput = torch.Tensor()
end

function SC:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.input_dim)
  end
  self.bias:normal(0, std)
  self.weight:normal(0, std)
  return self
end

function SC:resetStates()
  self.input_prev:set()
  self.input_prev_back:set()
end

function SC:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  local I, O, K = self.input_dim, self.output_dim, self.kernel_size
  
--  print(string.format('N=%d T=%d I=%d O=%d K=%d', N, T, I, O, K))
  
  local in_h = self.input_prev
  local output = self.output
  local weight = self.weight
  
  output:resize(N, T, O)
  if in_h:dim() == 0 then
    in_h:resize(N, K-1, I):zero()
  end
  output:copy(self.bias:reshape(1, 1, O):expand(N, T, O))
  
  for i = 1, K do
    if i < (T + 1) then
--      print(string.format('output[%d,%d]:baddmm(input[1,%d],weight[%d])', i, T, T-i+1, i))
      output[{{}, {i, T}, {}}]:baddbmm(input[{{}, {1, T-i+1}, {}}], weight[{{i}}]:expand(N, I, O))
    end
    if i > 1 then
      local e = math.max(0, i-1-T)
--      print(string.format('output[1, %d]:baddbmm(in_h[%d, %d],weight[%d])', i-e-1, 1-i, K-e-1, i))
      output[{{}, {1, i-e-1}, {}}]:baddbmm(in_h[{{}, {1-i, K-e-1}, {}}], weight[{{i}}]:expand(N, I, O))
    end
  end
  
  if self.train then
    self.input_prev_back:resizeAs(in_h):copy(in_h)
  end
  
  if K-1 <= T then
--    print(string.format('in_h[] = input[%d,-1]', 1-K))
    in_h:copy(input[{{}, {1-K, -1}, {}}])
  else
    for i = 1, K-T-1 do
--      print(string.format('in_h[] = in_h[%d]', i, i+T))
      in_h[{{}, i, {}}]:copy(in_h[{{}, i+T, {}}])
    end
--    print(string.format('in_h[%d,-1] = input[]', -T))
    in_h[{{}, {-T, -1}, {}}]:copy(input)
  end
  return output
end

function SC:updateGradInput(input, gradOutput)
  local N, T = input:size(1), input:size(2)
  local I, O, K = self.input_dim, self.output_dim, self.kernel_size

  local in_h = self.input_prev_back
  local output = self.output
  local weight = self.weight
  local gradInput = self.gradInput

  gradInput:resize(N, T, I):zero()
  
  for i = 1, K do
    if i < (T + 1) then
--      print(string.format('gradInput[1,%d]:baddbmm(gradOutput[%d,%d], weight[%d])', T-i+1, i, T, i))
      gradInput[{{}, {1, T-i+1}, {}}]:baddbmm(gradOutput[{{}, {i, T}, {}}], weight[{{i}}]:expand(N, I, O):transpose(2,3))
    end
  end
  
  return gradInput
end

function SC:accGradParameters(input, gradOutput, scale)
  local N, T = input:size(1), input:size(2)
  local I, O, K = self.input_dim, self.output_dim, self.kernel_size
  scale = scale or 1.0

  local in_h = self.input_prev_back
  local output = self.output
  local weight = self.weight
  local gradWeight = self.gradWeight
  local gradBias = self.gradBiasBuffer
  
  for i = 1, K do
    if i < (T + 1) then
--      print(string.format('gradWeight[%d]:addbmm(input[1,%d], output[%d,%d])', i, T-i+1, i, T))
      gradWeight[i]:addbmm(scale, input[{{}, {1, T-i+1}, {}}]:transpose(2,3), gradOutput[{{}, {i, T}, {}}])
    end
    if i > 1 then
      local e = math.max(0, i-1-T)
--      print(string.format('gradWeight[%d]:addbmm(in_h[%d,%d], output[1,%d])', i, 1-i, K-e-1, i-e-1))
      gradWeight[i]:addbmm(in_h[{{}, {1-i, K-e-1}, {}}]:transpose(2,3), gradOutput[{{}, {1, i-e-1}, {}}])
    end
  end
  
  gradBias:resize(O)
  gradBias:sum(gradOutput:reshape(N*T, O), 1)
  self.gradBias:add(gradBias)
end

function SC:clearState()
  self.output:set()
  self.input_prev:set()
  self.input_prev_back:set()
  self.gradInput:set()
  self.gradBiasBuffer:set()
end

function SC:__tostring__()
  return string.format('StatefulConvolution: %d=>%d, K=%d, D=%d', self.input_dim, self.output_dim, self.kernel_size, self.dilat)
end

function SC:set_batch_size(N)
  self.input_prev:resize(N, self.kernel_size-1, self.input_dim)
end

function SC:getState(n)
  return self.input_prev[n]
end

--function SC:parameters()
--  return {}, {}
--end
