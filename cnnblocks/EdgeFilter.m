classdef EdgeFilter < dagnn.Filter
% EDGEFILTER layer performing edge filtering function on edge maps.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2018. 

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = cnn_edgefilter(inputs{1}, [params{1}, params{2}, params{3}, params{4}]) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      [derInputs{1}, derParams_tmp] = cnn_edgefilter(inputs{1}, [params{1}, params{2}, params{3}, params{4}], derOutputs{1}) ;
      derParams{1} = derParams_tmp(1);
      derParams{2} = derParams_tmp(2);
      derParams{3} = derParams_tmp(3);
      derParams{4} = derParams_tmp(4);
    end

    function params = initParams(obj)
      params{1} = single(1);
      params{2} = single(0.5);
      params{3} = single(500);
      params{4} = single(0.1);
    end

    function obj = EdgeFilter(varargin)
      obj.load(varargin) ;
    end
  end
end
