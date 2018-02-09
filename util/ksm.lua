local ksm = {}

function ksm.make_mergeable(tensor)
  local ffi = require("ffi")
  if ffi.os ~= "Linux" then error("Not supported on this OS") end
  local storage = tensor:storage()
  local size = storage:elementSize() * storage:size()
  local ptr = storage:data()
  ffi.cdef[[
    int madvise(void *addr, size_t length, int advice);
  ]]
  local ptr_aligned = ffi.cast("char *", ptr) - 64 -- Works for large allocations. Will break sooner or later.
  local size_aligned = size - (size % 4096)

  ffi.C.madvise(ptr_aligned, size_aligned, 12) -- MADV_MERGEABLE
end

function ksm.make_parameters_mergeable(model)
  local tensors = model:parameters()
  for k,v in ipairs(tensors) do
    ksm.make_mergeable(v)
  end
end

return ksm
