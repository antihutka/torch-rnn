require 'LanguageModel'

local cmd = torch.CmdLine()
cmd:option('-i', 'cv/in.t7')
cmd:option('-o', 'cv/out.t7')
opt = cmd:parse(arg)

cp = torch.load(opt.i)
function rg(module)
  local t = torch.type(module)
  --print(t)
  if (t == 'nn.TemporalAdapter') then module.net:apply(rg) end
  module.gradWeight = nil
  module.gradBias = nil
end
cp.model.net:apply(rg)
torch.save(opt.o, cp)
