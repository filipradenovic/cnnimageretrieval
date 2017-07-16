classdef L2N < dagnn.Filter
% L2N layer that is L2-normalizing input vectors
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnnormalizelp(inputs{1}, [], 'p', 2, 'epsilon', 1e-6) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnnormalizelp(inputs{1}, derOutputs{1}, 'p', 2, 'epsilon', 1e-6);
      derParams = {} ;
    end

    function obj = L2N(varargin)
      obj.load(varargin) ;
    end
  end
end
