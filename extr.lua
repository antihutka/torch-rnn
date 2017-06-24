local utils = require 'util.utils'

local cmd = torch.CmdLine()

cmd:option('-out', 'valloss')
cmd:option('-json', 'cv/mora140-3-gridgru-1/_4000.json')
cmd:option('-avg', 1)
cmd:option('-load_prev', 1)
local opt = cmd:parse(arg)

local first_iter = 0
local val_num = 0
local val_sum = 0
local pv_extra = 0
local pv_last = -1

function pv(i, v)
  --print("pv(" .. i .. "," .. v .. ") last=" .. pv_last .. " extra=" .. pv_extra)

  if i + pv_extra < pv_last then
    --print("bad order!")
    pv_extra = pv_last
  end
  i = i + pv_extra

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
  pv_last = i
end

local jsons = {}

function load_jsons(name)
  --print('Loading json ' .. name)
  local jsn = utils.read_json(name)
  if opt.load_prev == 1 and jsn.opt.init_from and jsn.opt.init_from ~= '' then load_jsons(jsn.opt.init_from:gsub(".t7", ".json")) end
  --print('Loaded json ' .. name)
  table.insert(jsons, jsn)
end

--local jsn = utils.read_json(opt.json)
load_jsons(opt.json)

local ji,jsn
local iter = 1
for ji,jsn in ipairs(jsons) do
  --print('printing json ' .. ji)
  if opt.out == 'valloss' then
    local iter = jsn.val_loss_history_it
    local loss = jsn.val_loss_history
    local k,v,j
    for k,v in pairs(iter) do
      pv(v,loss[k])
    end
  elseif opt.out == 'trainloss' then
    local loss = jsn.train_loss_history
    for k,v in pairs(loss) do
      pv(iter,v)
      iter = iter + 1
    end
  end
end

pv(1e60, 0)
