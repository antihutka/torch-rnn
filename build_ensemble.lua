require 'LanguageModel'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-output', 'cv/ensemble.t7')
local opt = cmd:parse(arg)

local models = {}

while true do
  local line = io.read()
  if line == nil then break end
  print('Loading checkpoint ' .. line)
  cp = torch.load(line)
  table.insert(models, cp)
end

local new_net = nn.Sequential()
local ctable = nn.ConcatTable()
new_net:add(ctable)
local new_rnns = {}

for k,v in ipairs(models) do
  ctable:add(v.model.net)
  for l, w in ipairs(v.model.rnns) do
    table.insert(new_rnns, w)
  end
end

new_net:add(nn.CAddTable(true))
new_net:add(nn.MulConstant(1/#models, true))

local new_model = models[1]
new_model.model.net = new_net
new_model.model.rnns = new_rnns
--print(new_model.rnns)

function rg(module)
  local t = torch.type(module)
  --print(t)
  if (t == 'nn.TemporalAdapter') then module.net:apply(rg) end
  module.gradWeight = nil
  module.gradBias = nil
end
new_net:apply(rg)
print('saving to ' .. opt.output)
torch.save(opt.output, new_model)
