require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
require 'util.DataLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/tiny-shakespeare.h5')
cmd:option('-input_json', 'data/tiny-shakespeare.json')
cmd:option('-batch_size', 50)
cmd:option('-seq_length', 50)
cmd:option('-seq_offset', 0)
cmd:option('-shuffle_data', 0)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 0)
cmd:option('-history_depth', 0)
cmd:option('-rank', 64)
cmd:option('-zoneout', 0)
cmd:option('-zoneoutd', 0)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 2e-3)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)

-- Output options
cmd:option('-print_every', 1)
cmd:option('-checkpoint_every', 1000)
cmd:option('-checkpoint_name', 'cv/checkpoint')

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-gpu_opt', -2)

cmd:option('-swaprnn', 0)
cmd:option('-low_mem_dropout', 1)

local opt = cmd:parse(arg)


-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  -- Memory benchmarking is only supported in CUDA mode
  -- TODO: Time benchmarking is probably wrong in OpenCL mode.
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  print(string.format('Running with OpenCL on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
  print 'Running in CPU mode'
end


-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
end

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt_clone.idx_to_token = idx_to_token
local model = nil
local start_i = 0
if opt.init_from ~= '' then
  print('Initializing from ', opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  model = checkpoint.model:type(dtype)
  if opt.reset_iterations == 0 then
    start_i = checkpoint.i
  end
else
  model = nn.LanguageModel(opt_clone)
end
local params, grad_params, params_o, grad_params_o
local crit = nn.CrossEntropyCriterion():type(dtype)

local function set_model_type()
  model:type(dtype)
  params, grad_params = model:getParameters()
  if opt.swaprnn > 0 then model:swappable(opt.swaprnn) end
  if opt.gpu_opt == -2 then params_o, grad_params_o = params, grad_params end
end

set_model_type()

if opt.gpu_opt > -2 then
  params_o = torch.FloatTensor(params:size()):copy(params)
  grad_params_o = torch.FloatTensor(grad_params:size()):zero()
  if opt.gpu_opt > -1 then
    cutorch.withDevice(opt.gpu_opt + 1, function()
      params_o = params_o:cuda()
      grad_params_o = grad_params_o:cuda()
    end)
  end
end

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {}
local init_memory_usage, memory_usage = nil, {}

print('number of parameters: ' .. params:nElement())

if opt.memory_benchmark == 1 then
  -- This should only be enabled in GPU mode
  assert(cutorch)
  cutorch.synchronize()
  local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
  init_memory_usage = total - free
end

-- Loss function that we pass to an optim method
local function f(w)
  assert(w == params or w == params_o)
  grad_params:zero()

  -- Get a minibatch and run the model forward, maybe timing it
  local timer
  local x, y, z = loader:nextBatch('train')
  x, y = x:type(dtype), y:type(dtype)
  if z then z:type(dtype) end
  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end

  if z then
    model:resetStates()
    model:forward(z)
  end

  local scores = model:forward(x)

  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)

  local time_f
  if timer then
    time_f = timer:time().real
  end

  -- Run the Criterion and model backward to compute gradients, maybe timing it
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)
  if timer then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print(string.format('Forward / Backward pass took %.4f (%.4f/%.4f)', time, time_f, time - time_f))
    table.insert(forward_backward_times, time)
  end

  -- Maybe record memory usage
  if opt.memory_benchmark == 1 then
    assert(cutorch)
    if cutorch then cutorch.synchronize() end
    local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
    local memory_used = total - free - init_memory_usage
    local memory_used_mb = memory_used / 1024 / 1024
    print(string.format('Using %dMB of memory', memory_used_mb))
    table.insert(memory_usage, memory_used)
  end

  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  if opt.gpu_opt > -2 then
    grad_params_o:copy(grad_params)
    if opt.gpu_opt > -1 then cutorch.setDevice(opt.gpu_opt + 1) end
  end
  return loss, grad_params_o
end

-- Train the model!
local optim_config = {learningRate = opt.learning_rate}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
local avg_loss = 0
local trend = 0
local iteration_timer = torch.Timer()
local val_loss_best, val_loss_best_at
local val_loss_last = 0
model:training()
for i = start_i + 1, num_iterations do
  local epoch = math.floor(i / num_train) + 1

  -- Check if we are at the end of an epoch
  if i % num_train == 0 then
    model:resetStates() -- Reset hidden states

    -- Maybe decay learning rate
    if epoch % opt.lr_decay_every == 0 then
      local old_lr = optim_config.learningRate
      -- optim_config = {learningRate = old_lr * opt.lr_decay_factor}
      optim_config.learningRate = old_lr * opt.lr_decay_factor
      print('Changing learning rate to ' .. optim_config.learningRate)
    end
  end

  -- Take a gradient step and maybe print
  -- Note that adam returns a singleton array of losses
  local _, loss = optim.adam(f, params_o, optim_config)
  if opt.gpu_opt > -2 then
    params:copy(params_o)
    if opt.gpu_opt > -1 then cutorch.setDevice(opt.gpu + 1) end
  end

  table.insert(train_loss_history, loss[1])
  if avg_loss == 0 then avg_loss = loss[1] end
  local avg_loss_old = avg_loss
  avg_loss = avg_loss * 0.995 + loss[1] * 0.005
  trend = trend * 0.995 + (avg_loss - avg_loss_old) * 0.005
  if opt.print_every > 0 and i % opt.print_every == 0 then
    local iter_time = iteration_timer:time().real / opt.print_every
    iteration_timer:reset()
    local float_epoch = i / num_train + 1
    local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f, avg_loss = %f, delta = %9.6f, trend = %10.7f, time = %.2f, tps = %.1f'
    local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1], avg_loss, avg_loss - avg_loss_old, trend, iter_time, opt.batch_size * opt.seq_length / iter_time}
    print(string.format(unpack(args)))
  end

  -- Maybe save a checkpoint
  local check_every = opt.checkpoint_every
  if (check_every > 0 and i % check_every == 0) or i == num_iterations then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    model:evaluate()
    model:resetStates()
    local num_val = loader.split_sizes['val']
    local val_loss = 0
    for j = 1, num_val do
      local xv, yv = loader:nextBatch('val')
      xv = xv:type(dtype)
      yv = yv:type(dtype):view(N * T)
      local scores = model:forward(xv):view(N * T, -1)
      val_loss = val_loss + crit:forward(scores, yv)
    end
    val_loss = val_loss / num_val
    if (not val_loss_best) or (val_loss_best > val_loss) then
      val_loss_best = val_loss
      val_loss_best_at = i
    end
    print(string.format("val_loss = %.6f [%.6f], best = %.6f @ %d [%d]", val_loss, val_loss - val_loss_last, val_loss_best, val_loss_best_at, i - val_loss_best_at))
    val_loss_last = val_loss
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
    model:resetStates()
    model:training()

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
      forward_backward_times = forward_backward_times,
      memory_usage = memory_usage,
      i = i
    }
    local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
    -- Make sure the output directory exists before we try to write it
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)

    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
    model:clearState()
    grad_params:zero()
    model:float()
    checkpoint.model = model
    local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
    paths.mkdir(paths.dirname(filename))
    torch.save(filename, checkpoint)
    params, grad_params = nil, nil
    collectgarbage()
    set_model_type()
    collectgarbage()
  end
end
