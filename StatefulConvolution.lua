local SC, parent = torch.class('nn.StatefulConvolution', 'nn.Module')

-- N - batches, T - time, I - input dim, O - output dim, D - dilat
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
  local I, O, K, D = self.input_dim, self.output_dim, self.kernel_size, self.dilat
  local H = (K-1)*D
  
--  print(string.format('N=%d T=%d I=%d O=%d K=%d', N, T, I, O, K))
  
  local in_h = self.input_prev
  local output = self.output
  local weight = self.weight
  
  output:resize(N, T, O)
  if in_h:dim() == 0 then
    in_h:resize(N, H, I):zero()
  end
  output:copy(self.bias:reshape(1, 1, O):expand(N, T, O))
  
  
  for i = 1, K do
    local t = (i-1)*D
    if t < T then
      output[{{}, {t+1, T}, {}}]:baddbmm(input[{{}, {1, T-t}, {}}], weight[{{i}}]:expand(N, I, O))
    end
    if i > 1 then
      local e = math.max(0, t-T)
      output[{{}, {1, t-e}, {}}]:baddbmm(in_h[{{}, {-t, H-e}, {}}], weight[{{i}}]:expand(N, I, O))
    end
  end
  
  if self.train then
    self.input_prev_back:resizeAs(in_h):copy(in_h)
  end
  
  if H <= T then
    in_h:copy(input[{{}, {-H, -1}, {}}])
  else
    for i = 1, H-T do
      in_h[{{}, i, {}}]:copy(in_h[{{}, i+T, {}}])
    end
    in_h[{{}, {-T, -1}, {}}]:copy(input)
  end
  return output
end

function SC:updateGradInput(input, gradOutput)
  local N, T = input:size(1), input:size(2)
  local I, O, K, D = self.input_dim, self.output_dim, self.kernel_size, self.dilat

  local in_h = self.input_prev_back
  local output = self.output
  local weight = self.weight
  local gradInput = self.gradInput

  gradInput:resize(N, T, I):zero()
  
  for i = 1, K do
    local t = (i-1)*D
    if t < T then
      gradInput[{{}, {1, T-t}, {}}]:baddbmm(gradOutput[{{}, {t+1, T}, {}}], weight[{{i}}]:expand(N, I, O):transpose(2,3))
    end
  end
  
  return gradInput
end

function SC:accGradParameters(input, gradOutput, scale)
  local N, T = input:size(1), input:size(2)
  local I, O, K, D = self.input_dim, self.output_dim, self.kernel_size, self.dilat
  local H = (K-1)*D
  scale = scale or 1.0

  local in_h = self.input_prev_back
  local output = self.output
  local weight = self.weight
  local gradWeight = self.gradWeight
  local gradBias = self.gradBiasBuffer
  
  for i = 1, K do
    local t = (i-1)*D
    if i < (T + 1) then
      gradWeight[i]:addbmm(scale, input[{{}, {1, T-t}, {}}]:transpose(2,3), gradOutput[{{}, {t+1, T}, {}}])
    end
    if i > 1 then
      local e = math.max(0, i-1-T)
      gradWeight[i]:addbmm(in_h[{{}, {-t, H-e}, {}}]:transpose(2,3), gradOutput[{{}, {1, t-e}, {}}])
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
