require 'torch'
require 'nn'
require 'LanguageModel'

local cmd = torch.CmdLine()
cmd:option('-i', 'cv/in.t7')
cmd:option('-o', 'cv/out.t7')
cmd:option('-j', '')
opt = cmd:parse(arg)

cp = torch.load(opt.i)

local known_storages = {}

function maptensor(t)
  local storage = t:storage()
  local ptr = torch.pointer(storage)
  local stride = t:stride()
  local offset = t:storageOffset()
  local size = t:size()
  if not known_storages[ptr] then
    local storageidx = #known_storages
    local filename = opt.o .. '.' .. storageidx
    print(string.format('Found storage id=%d size=%d file=%s', storageidx, storage:size(), filename))
    local newstorage = torch.FloatStorage(filename, true, storage:size())
    newstorage:copy(storage)
    known_storages[ptr] = storageidx
  end
  local storageidx = known_storages[ptr]
  return {storage=storageidx, stride=stride, offset=offset, size=size}
end

function rg(module)
  local t = torch.type(module)
  --print(t)
  if (t == 'nn.TemporalAdapter') then module.net:apply(rg) end

  if module.weight then
    module.weight = maptensor(module.weight)
  end

  if module.bias then
    module.bias = maptensor(module.bias)
  end

end

cp.model.net:apply(rg)
cp.is_mapped = true
torch.save(opt.o, cp)

function jsonize(t)
  local j = {}
  j.storage = t.storage
  j.offset = t.offset - 1
  j.size = {}
  j.stride = {}
  for i = 1, t.size:size() do
    table.insert(j.size, t.size[i])
    table.insert(j.stride, t.stride[i])
  end
  return j
end

if opt.j ~= '' then
  local util = require 'util.utils'
  local jdata = {}
  jdata.idx_to_token = cp.model.idx_to_token
  jdata.token_to_idx = cp.model.token_to_idx
  jdata.layers = {}
  for i,m in ipairs(cp.model.net.modules) do
    local mtype = torch.type(m)
    print(string.format("found module %d: %s", i, mtype))
    local jm = {}
    if mtype == 'nn.LookupTable' then
      jm.type = 'LookupTable'
      jm.weight = jsonize(m.weight)
    elseif mtype == 'nn.GRIDGRU' then
      jm.type = 'GRIDGRU'
      jm.weight = jsonize(m.weight)
      jm.bias = jsonize(m.bias)
      jm.zoneout_p = m.zoneout_prob
      jm.zoneout_pd = m.zoneout_probd
      jm.input_dim = m.input_dim
      jm.hidden_dim = m.hidden_dim
    elseif mtype == 'nn.Dropout' then
      jm.type = 'Dropout'
      jm.p = m.p
    elseif mtype == 'nn.TemporalAdapter' then
      jm.type = 'Linear'
      jm.weight = jsonize(m.net.modules[2].weight)
      jm.bias = jsonize(m.net.modules[2].bias)
    else
      error("unknown type")
    end
    table.insert(jdata.layers, jm)
  end
  util.write_json(opt.j, jdata)
end
