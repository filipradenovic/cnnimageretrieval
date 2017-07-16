classdef SPOC < dagnn.Filter
% SPOC layer that computes global Average Pooling of convolution activations.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnpool(inputs{1}, [size(inputs{1},1), size(inputs{1},2)], 'method', 'avg', 'cuDNN');
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnpool(inputs{1}, [size(inputs{1},1), size(inputs{1},2)], derOutputs{1}, 'method', 'avg', 'cuDNN');
      derParams = {};
    end

    function obj = SPOC(varargin)
      obj.load(varargin);
    end
  end
end
