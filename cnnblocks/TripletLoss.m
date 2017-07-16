classdef TripletLoss < dagnn.Layer
% TRIPLETLOSS layer that computes triplet loss for a batch of images:
%   Q query tuples, each packed in the form of (q,p,n1,..nN)
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

  properties
    margin = 0.1
  end
  
  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = cnn_tripletloss(inputs{1}, inputs{2}, obj.margin);
      nq = sum(gather(inputs{2}) == -1); % number of query tuples in this batch
      n = obj.numAveraged; % number of query tuples done before this batch
      m = n + nq; % number of query tuples done so far
      obj.average = (n * obj.average + gather(outputs{1})) / m;
      obj.numAveraged = m;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [derInputs{1}] = cnn_tripletloss(inputs{1}, inputs{2}, obj.margin, derOutputs{1});
      derInputs{2} = [];
      derParams = {};
    end

    function reset(obj)
      obj.average = 0;
      obj.numAveraged = 0;
    end

    function obj = TripletLoss(varargin)
      obj.load(varargin);
    end
  end
end
