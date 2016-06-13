require 'torch'
require 'nn'
require 'rnn'

neuralconvo = {}


torch.include('neuralconvo', 'seq2seq.lua')

return neuralconvo