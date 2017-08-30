require 'torch'
require 'nn'

local lu = require 'util.layer_utils'

local layer, parent = torch.class('nn.GRIDGRU', 'nn.Module')

--[[
Adapted from Grid LSTM : http://arxiv.org/abs/1507.01526
--]]

function layer:__init(input_dim, hidden_dim)
  parent.__init(self)

  local D, H = input_dim, hidden_dim
  self.input_dim, self.hidden_dim = D, H

  self.weight = torch.Tensor(D + H, 3 * H + 3 * D)
  self.gradWeight = torch.Tensor(D + H, 3 * H + 3 * D):zero()
  self.bias = torch.Tensor(3 * H + 3 * D)
  self.gradBias = torch.Tensor(3 * H + 3 * D):zero()
  --self.weightd = torch.Tensor(D + H, 3 * D)
  --self.gradWeightd = torch.Tensor(D + H, 3 * D):zero()
  --self.biasd = torch.Tensor(3 * D)
  --self.gradBiasd = torch.Tensor(3 * D):zero()
  self:reset()

  self.cell = torch.Tensor()    -- This will be (N, T, H)
  self.gates = torch.Tensor()   -- This will be (N, T, 3H)
  self.gatesd = torch.Tensor()   -- This will be (N, T, 3H)
  self.buffer1 = torch.Tensor() -- This will be (N, H)
  self.buffer2 = torch.Tensor() -- This will be (N, H)
  self.buffer3 = torch.Tensor() -- This will be (H,)
  self.grad_a_buffer = torch.Tensor() -- This will be (N, 3H)
  self.buffer1d = torch.Tensor() -- This will be (N, D)
  self.buffer2d = torch.Tensor() -- This will be (N, D)
  self.buffer3d= torch.Tensor() -- This will be (D,)
  self.grad_a_bufferd = torch.Tensor() -- This will be (N, 3D)
  self.h0 = torch.Tensor()
  self.remember_states = false
  self.grad_h0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.gradInput = {self.grad_h0, self.grad_x}
end


function layer:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  --self.bias:zero()
  self.bias:normal(0,std)
  self.weight:normal(0, std)
  return self
end


function layer:resetStates()
  self.h0 = self.h0.new()
end


local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end


function layer:_unpack_input(input)
  local h0, x = nil, nil

  if torch.type(input) == 'table' and #input == 2 then
    h0, x = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return h0, x
end


function layer:_get_sizes(input, gradOutput)
  local h0, x = self:_unpack_input(input)
  local N, T = x:size(1), x:size(2)
  local H, D = self.hidden_dim, self.input_dim
  check_dims(x, {N, T, D})
  if h0 then
    check_dims(h0, {N, H})
  end

  if gradOutput then
    check_dims(gradOutput, {N, T, D})
  end
  return N, T, D, H
end

function layer:_split_weights(w)
  local H, D = self.hidden_dim, self.input_dim
  local Wx = w[{{1, D}}]
  local Wh = w[{{D + 1, D + H}}]
  local Wxt = Wx[{{},{1, 3 * H}}]
  local Whtg = Wh[{{},{1, 2 * H}}]
  local Whtc = Wh[{{},{2 * H + 1, 3 * H}}]
  local Wxd = Wx[{{},{3 * H + 1, 3 * H + 3 * D}}]
  local Whd = Wh[{{},{3 * H + 1, 3 * H + 3 * D}}]
  return Wxt, Whtg, Whtc, Wxd, Whd
end

--[[
Input:
- h0: Initial hidden state, (N, H)
- x: Input sequence, (N, T, D)

Output:
- h: Sequence of hidden states, (N, T, D)
--]]


function layer:updateOutput(input)
  local h0, x = self:_unpack_input(input)
  local N, T, D, H = self:_get_sizes(input)

  self._return_grad_h0 = (h0 ~= nil)

  self:swapin()

  if not h0 then
    h0 = self.h0
    if h0:nElement() == 0 or not self.remember_states then
      h0:resize(N, H):zero()
    elseif self.remember_states then
      local prev_N, prev_T = self.cell:size(1), self.cell:size(2)
      assert(prev_N == N, 'batch sizes must be the same to remember states')
      h0:copy(self.cell[{{}, prev_T}])
    end
  end

  local bias_expand = self.bias:view(1, 3 * H + 3 * D):expand(N, 3 * H + 3 * D)
  local bias_expand_nt = self.bias:view(1, 3 * H + 3 * D):expand(N * T, 3 * H + 3 * D)
  local Wxt, Whtg, Whtc, Wxd, Whd = self:_split_weights(self.weight)
  local bias_expandt_nt = bias_expand_nt[{{},{1, 3 * H}}]
  local bias_expandd = bias_expand[{{},{3 * H + 1, 3 * H + 3 * D}}]
  local bias_expandd_b = nn.utils.addSingletonDimension(bias_expandd, 1):expand(T, N, 3 * D)

  local h, ht = self.output, self.cell
  h:resize(N, T, D):zero()
  ht:resize(N, T, H):zero()
  local prev_ht = h0
  self.gates:resize(N, T, 3 * H)
  self.gatesd:resize(N, T, 3 * D):copy(bias_expandd_b)

  local gates_nt = self.gates:view(N * T, 3 * H)
  local gatesd_nt = self.gatesd:view(N * T, 3 * D)
  local x_nt = x:view(N * T, D)
  local ht_nt = ht:view(N * T, H)
  local h_nt = h:view(N * T, D)

  gates_nt:addmm(bias_expandt_nt, x_nt, Wxt)
  gatesd_nt[{{}, {1, 2 * D}}]:addmm(x_nt, Wxd[{{}, {1, 2 * D}}])

  for t = 1, T do
    local next_ht = ht[{{}, t}]
    local cur_gates = self.gates[{{}, t}]
    local cur_gates_g = cur_gates[{{}, {1, 2 * H}}]

    cur_gates_g:addmm(prev_ht, Whtg)
    cur_gates_g:sigmoid()

    local u = cur_gates[{{}, {1, H}}] --update gate : u = sig(Wx * x + Wh * prev_h + b)
    local r = cur_gates[{{}, {H + 1, 2 * H}}] --reset gate : r = sig(Wx * x + Wh * prev_h + b)
    next_ht:cmul(r, prev_ht) --temporary buffer : r . prev_h
    local hc = cur_gates[{{}, {2 * H + 1, 3 * H}}]
    hc:addmm(next_ht, Whtc) -- hc += Wh * r . prev_h
    hc:tanh() --hidden candidate : hc = tanh(Wx * x + Wh * r . prev_h + b)
    next_ht:addcmul(prev_ht,-1, u, prev_ht)
    next_ht:addcmul(u,hc)  --next_h = (1-u) . prev_h + u . hc
    prev_ht = next_ht
  end

  gatesd_nt:addmm(ht_nt, Whd)
  self.gatesd[{{}, {}, {1, 2 * D}}]:sigmoid()
  local ud_b = self.gatesd[{{}, {}, {1, D}}]
  local rd_b = self.gatesd[{{}, {}, {D + 1, 2 * D}}]
  local hcd_b = gatesd_nt[{{}, {2 * D + 1, 3 * D}}]
  h:cmul(rd_b, x)
  hcd_b:addmm(h_nt, Wxd[{{}, {2 * D + 1, 3 * D}}])
  hcd_b:tanh()
  h:addcmul(x, -1, ud_b, x)
  h:addcmul(ud_b, hcd_b)
  self:swapout()

  return self.output
end

function layer:backward(input, gradOutput, scale)
  scale = scale or 1.0
  local h0, x = self:_unpack_input(input)

  self:swapin()

  if not h0 then h0 = self.h0 end

  local grad_h0_tb, grad_x = self.grad_h0, self.grad_x
  local ht = self.cell
  local grad_h = gradOutput

  local TB = 8 -- number of timesteps to batch operations for

  local N, T, D, H = self:_get_sizes(input, gradOutput)
  local Wxt, Whtg, Whtc, Wxd, Whd = self:_split_weights(self.weight)
  local grad_Wxt, grad_Whtg, grad_Whtc, grad_Wxd, grad_Whd = self:_split_weights(self.gradWeight)

  local grad_b = self.gradBias
  local grad_bt = grad_b[{{1, 3 * H}}]
  local grad_bd = grad_b[{{3 * H + 1, 3 * H + 3 * D}}]

  grad_h0_tb:resize(TB, N, H)
  grad_x:resize(T, N, D):zero()
  local grad_next_h = self.buffer1:resizeAs(h0):zero()
  local temp_buffer_tb = self.buffer2:resize(TB, N, H)
  local temp_buffer = temp_buffer_tb[1]
  local grad_a_sum = self.buffer3:resize(1,3*H):zero()
  local grad_next_hd_tb = self.buffer1d:resize(TB, N, D):zero()
  local temp_bufferd_tb = self.buffer2d:resize(TB, N, D)
  local grad_a_sumd = self.buffer3d:resize(1,3*D):zero()

  local grad_ad_tb = self.grad_a_bufferd:resize(TB, N, 3 * D):zero()

  local grad_a_tb = self.grad_a_buffer:resize(TB, N, 3 * H)

  for t = T, 1, -1 do
    local prev_h
    if t == 1 then
      prev_h = h0
    else
      prev_h = ht[{{}, t - 1}]
    end

    local TBi = (t-1) % TB
    local grad_h0 = grad_h0_tb[TBi + 1]

    local grad_a = grad_a_tb[TBi + 1]
    local grad_au = grad_a[{{}, {1, H}}]
    local grad_ar = grad_a[{{}, {H + 1, 2 * H}}]
    local grad_ahc = grad_a[{{}, {2 * H + 1, 3 * H}}]

    if TBi == TB - 1 or t == T then
      local TBl = TBi + 1
      local tfirst = t - TBi

      local grad_ad_t = grad_ad_tb[{{1, TBl}, {}}]:transpose(1,2)
      local grad_aud_t = grad_ad_t[{{}, {}, {1, D}}]
      local grad_ard_t = grad_ad_t[{{}, {}, {D + 1, 2 * D}}]
      local grad_ahcd_t = grad_ad_t[{{}, {}, {2 * D + 1, 3 * D}}]

      local grad_ad_tn = grad_ad_tb[{{1, TBl}}]:view(TBl*N, 3*D)
      local grad_aud_tn = grad_ad_tn[{{}, {1, D}}]
      local grad_ahcd_tn = grad_ad_tn[{{}, {2 * D + 1, 3 * D}}]
      local grad_h0_tn = grad_h0_tb[{{1, TBl}}]:view(TBl*N, H)

      local temp_bufferd_t = temp_bufferd_tb[{{1, TBl}}]:transpose(1,2)
      local temp_bufferd_tn = temp_bufferd_tb[{{1, TBl}}]:view(TBl * N, D)

      local ud_t = self.gatesd[{{}, {tfirst, t}, {1, D}}]
      local rd_t = self.gatesd[{{}, {tfirst, t}, {D + 1, 2 * D}}]
      local hcd_t = self.gatesd[{{}, {tfirst, t}, {2 * D + 1, 3 * D}}]
      local x_t = x[{{}, {tfirst, t}}]
      local grad_h_t = grad_h[{{}, {tfirst, t}, {}}]

      grad_aud_t:cmul(grad_h_t, ud_t)
      lu.tanh_gradient(grad_ahcd_t, hcd_t, grad_aud_t)
      grad_aud_tn:mm(grad_ahcd_tn, Wxd[{{}, {2 * D + 1, 3 * D}}]:t())
      grad_aud_t:cmul(x_t)
      lu.sigmoid_gradient(grad_ard_t, rd_t, grad_aud_t)
      temp_bufferd_t:add(hcd_t, -1, x_t)
      lu.sigmoid_gradient(grad_aud_t, ud_t, grad_h_t)
      grad_aud_t:cmul(temp_bufferd_t)
      grad_h0_tn:mm(grad_ad_tn, Whd:t())
      grad_Whd:addbmm(scale, ht[{{}, {tfirst, t}}]:transpose(1,2):transpose(2,3), grad_ad_tb[{{1, TBl}}])
      grad_Wxd[{{}, {1, 2 * D}}]:addbmm(scale, x_t:transpose(1,2):transpose(2,3), grad_ad_tb[{{1, TBl}, {}, {1, 2 * D}}])
      grad_a_sumd:sum(grad_ad_tn, 1)
      grad_bd:add(scale, grad_a_sumd)
      temp_bufferd_t:cmul(x_t, rd_t)
      grad_Wxd[{{}, {2 * D + 1, 3 * D}}]:addbmm(scale, temp_bufferd_t:transpose(1,2):transpose(2,3), grad_ad_tb[{{1, TBl}, {}, {2 * D + 1, 3 * D}}])
      temp_bufferd_tn:mm(grad_ahcd_tn, Wxd[{{}, {2 * D + 1, 3 * D}}]:t())
      temp_bufferd_t:cmul(rd_t)
    end

    local u = self.gates[{{}, t, {1, H}}]
    local r = self.gates[{{}, t, {H + 1, 2 * H}}]
    local hc = self.gates[{{}, t, {2 * H + 1, 3 * H}}]

    -- We will use grad_au as temporary buffer
    -- to compute grad_ahc.

    grad_next_h:add(grad_h0)
    local grad_hc = grad_au:cmul(grad_next_h, u)
    lu.tanh_gradient(grad_ahc, hc, grad_hc)
    local grad_r = grad_au:mm(grad_ahc, Whtc:t() ):cmul(prev_h)
    lu.sigmoid_gradient(grad_ar, r, grad_r)

    temp_buffer:add(hc, -1, prev_h)
    lu.sigmoid_gradient(grad_au, u, grad_next_h)
    grad_au:cmul(temp_buffer)

    grad_next_h:addcmul(-1, u, grad_next_h)
    grad_next_h:addmm(grad_a[{{}, {1, 2 * H}}], Whtg:t())
    temp_buffer:mm(grad_a[{{}, {2 * H + 1, 3 * H}}], Whtc:t()):cmul(r)
    grad_next_h:add(temp_buffer)

    if TBi == 0 then
      local tlast = t + TB - 1
      if tlast > T then tlast = T end
      local TBl = tlast - t + 1

      local grad_h_tb = grad_h[{{}, {t, tlast}}]
      local grad_next_hd_t = grad_next_hd_tb[{{1, TBl}}]
      local grad_a_t = grad_a_tb[{{1, TBl}}]
      local grad_a_tn = grad_a_t:view(TBl * N, 3 * H)
      local temp_buffer_t = temp_buffer_tb[{{1, TBl}}]
      local r_t = self.gates[{{}, {t, tlast}, {H + 1, 2 * H}}]

      grad_next_hd_t:transpose(1,2):addcmul(grad_h_tb, -1, self.gatesd[{{}, {t, tlast}, {1, D}}], grad_h_tb)
      grad_next_hd_t:view(TBl * N, D):addmm(grad_ad_tb[{{1, TBl}}]:view(TBl * N, 3 * D)[{{}, {1, 2 * D}}], Wxd[{{}, {1, 2 * D}}]:t())
      grad_next_hd_t:add(temp_bufferd_tb[{{1, TBl}}])
      grad_x[{{t, tlast}, {}}]:view(TBl * N, D):addmm(grad_next_hd_t:view(TBl * N, D), grad_a_tn, Wxt:t())
      grad_Wxt:addbmm(scale, x[{{}, {t, tlast}}]:transpose(1,2):transpose(2,3), grad_a_t)
      grad_a_sum:sum(grad_a_tn, 1)
      grad_bt:add(scale, grad_a_sum)
      if t > 1 then
        grad_Whtg:addbmm(scale, ht[{{}, {t-1, tlast-1}}]:transpose(1,2):transpose(2,3), grad_a_t[{{}, {}, {1, 2 * H}}])
        temp_buffer_t:transpose(1,2):cmul(ht[{{}, {t-1, tlast-1}}], r_t)
      else
        grad_Whtg:addbmm(scale, ht[{{}, {t, tlast-1}}]:transpose(1,2):transpose(2,3), grad_a_t[{{2, TBl}, {}, {1, 2 * H}}])
        grad_Whtg:addmm(scale, h0:t(), grad_a[{{}, {1, 2 * H}}])
        temp_buffer_t[{{2, TBl}}]:transpose(1,2):cmul(ht[{{}, {t, tlast-1}}], r_t[{{}, {2, TBl}}])
        temp_buffer_t[{1}]:cmul(h0, r_t[{{}, 1}])
      end
      grad_Whtc:addbmm(scale, temp_buffer_t:transpose(2,3), grad_a_tb[{{1, TBl}, {}, {2 * H + 1, 3 * H}}])
    end
  end

  if self._return_grad_h0 then
    grad_h0_tb[1]:copy(grad_next_h)
    self.gradInput = {self.grad_h0[1], self.grad_x:transpose(1,2)}
  else
    self.gradInput = self.grad_x:transpose(1,2)
  end

  self:swapout()

  return self.gradInput
end


function layer:updateGradInput(input, gradOutput)
  self:backward(input, gradOutput, 0)
end


function layer:accGradParameters(input, gradOutput, scale)
  self:backward(input, gradOutput, scale)
end

function layer:clearState()
  if self.swap then
    self.cell_sw:set()
    self.gates_sw:set()
    self.gatesd_sw:set()
  else
    self.cell:set()
    self.gates:set()
    self.gatesd:set()
  end
  self.buffer1:set()
  self.buffer2:set()
  self.buffer3:set()
  self.grad_a_buffer:set()
  self.buffer1d:set()
  self.buffer2d:set()
  self.buffer3d:set()
  self.grad_a_bufferd:set()

  self.grad_h0:set()
  self.grad_x:set()
  self.output:set()
end

function layer:__tostring__()
  return 'nn.GRIDGRU: ' .. self.input_dim .. 'x' .. self.hidden_dim
end

local swappable_tensors = { "cell", "gates", "gatesd" }

function layer:swapin()
  if not self.swapped or not self.swap then return end
  self.swapped = false
  for k,v in ipairs(swappable_tensors) do
    local vsw = v .. '_sw'
    --print('old size:' .. self[v]:numel())
    self[v]:resize(self[vsw]:size()):copyAsync(self[vsw])
  end
end

function layer:swapout()
  if self.swapped or not self.swap then return end
  self.swapped = true
  for k,v in ipairs(swappable_tensors) do
    local vsw = v .. '_sw'
    self[vsw]:resize(self[v]:size()):copyAsync(self[v])
  end
end

function layer:swappable(with)
  for k,v in ipairs(swappable_tensors) do
    local vsw = v .. '_sw'
    --self[vsw] = torch.FloatTensor()
    self[vsw] = cutorch.createCudaHostTensor(1)
    self[vsw]:resize(self[v]:size()):copy(self[v])
  end
  self.swap = true
  self.swapped = true
  if with then
    assert(with.swap)
    for k,v in ipairs(swappable_tensors) do
      self[v] = with[v]
    end
  end
end

function layer:type(type, tensorCache)
  if self.swap then
    for k,v in ipairs(swappable_tensors) do
      local vsw = v .. '_sw'
      self[v] = self[vsw]
      self[vsw] = nil
    end
    self.swapped = false
    self.swap = false
  end
  parent.type(self, type, tensorCache)
end

function layer:setBatchSize(N)
  local H = self.hidden_dim
  local T = self.cell:size(2)
  self.cell:resize(N, T, H):zero()
  self.h0:resize(N, H):zero()
end

function layer:getState(n)
  local T = self.cell:size(2)
  return self.cell[{n, T}]
end
