require 'torch'
require 'nn'
require 'LanguageModel'

local cmd = torch.CmdLine()
cmd:option('-i', 'cv/in.t7')
cmd:option('-o', 'cv/out.t7')
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
