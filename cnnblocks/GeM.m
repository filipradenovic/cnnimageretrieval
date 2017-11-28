classdef GeM < dagnn.Filter
% GEM layer that computes global Generalized Mean Pooling of convolution activations.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = cnn_gem(inputs{1}, params{1});
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      [derInputs{1}, derParams{1}] = cnn_gem(inputs{1}, params{1}, derOutputs{1});
    end

    function params = initParams(obj)
      params{1} = ones(obj.size,'single') * 3;
    end

    function obj = GeM(varargin)
      obj.load(varargin) ;
    end
  end
end
