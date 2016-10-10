local utils = require 'util.utils'

local cmd = torch.CmdLine()

cmd:option('-out', 'valloss')
cmd:option('-json', 'cv/mora140-3-gridgru-1/_4000.json')
cmd:option('-avg', 1)

local opt = cmd:parse(arg)
local jsn = utils.read_json(opt.json)

local first_iter = 0
local val_num = 0
local val_sum = 0

function pv(i, v)
  --print("pv(" .. i .. "," .. v .. ")")
  if i + 1 > first_iter + opt.avg and val_num > 0 then
    print(first_iter .. "\t" .. (val_sum / val_num))
    first_iter = 0
    val_num = 0
    val_sum = 0
  end
  if first_iter == 0 then first_iter = i end
  if i + 1 <= first_iter + opt.avg then
    val_num = val_num + 1
    val_sum = val_sum + v
  end
end

if opt.out == 'valloss' then
  local iter = jsn.val_loss_history_it
  local loss = jsn.val_loss_history
  local k,v
  for k,v in pairs(iter) do
    pv(v,loss[k])
  end
elseif opt.out == 'trainloss' then
  local loss = jsn.train_loss_history
  for k,v in pairs(loss) do
    pv(k,v)
  end
end

pv(1e60, 0)
