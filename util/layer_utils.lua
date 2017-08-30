layer_utils = {}

function layer_utils.sigmoid_gradient(gradInput, output, gradOutput)
  gradInput.THNN.Sigmoid_updateGradInput(nil, gradOutput:cdata(), gradInput:cdata(), output:cdata())
end

function layer_utils.tanh_gradient(gradInput, output, gradOutput)
  gradInput.THNN.Tanh_updateGradInput(nil, gradOutput:cdata(), gradInput:cdata(), output:cdata())
end

return layer_utils
