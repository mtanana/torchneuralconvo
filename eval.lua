require 'neuralconvo'
require 'util.Tester'
local tokenizer = require "tokenizer"
local list = require "pl.List"
require 'nn'
local WordSplitLMMinibatchLoader = require 'util.WordSplitLMMinibatchLoader'


local options = {}

if loader == nil then
  cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('--cuda', false, 'use CUDA. Training must be done on CUDA')
  cmd:option('--debug', false, 'show debug info')
  cmd:option('--dataset', "model.t7", 'show debug info')
  cmd:option('--vocablocation', "data/opensubssmall/vocabwords.t7", 'show debug info')
  cmd:text()
  options = cmd:parse(arg)


  -- Enabled CUDA
  if options.cuda then
    require 'cutorch'
    require 'cunn'
  end

  -- Data
  loader = WordSplitLMMinibatchLoader.createFromJustVocab(options.vocablocation)

end

if model == nil then
  print("-- Loading model")
  model = torch.load("data/"..options.dataset)
end



function say(text)
  print(getResponse(text,loader,model,options.debug))
end

repeat
  io.write("Ask: ")
  io.flush()
  answer=io.read()

  io.write(say(answer))

until answer=="end"
