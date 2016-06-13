require 'neuralconvo'
require 'util.Tester'
local tokenizer = require "tokenizer"
local list = require "pl.List"
local options = {}

if dataset == nil then
  cmd = torch.CmdLine()
  cmd:text('Options:')
  cmd:option('--cuda', true, 'use CUDA. Training must be done on CUDA')
  cmd:option('--debug', true, 'show debug info')
  cmd:option('--dataset', "model.t7", 'show debug info')
  cmd:text()
  options = cmd:parse(arg)

  -- Data
  dataset = neuralconvo.DataSet()

  -- Enabled CUDA
  if options.cuda then
    require 'cutorch'
    require 'cunn'
  end
end

if model == nil then
  print("-- Loading model")
  model = torch.load("data/"..options.dataset)
end



function say(text)
  print(getResponse(text,dataset,model,true))
end

repeat
  io.write("Ask: ")
  io.flush()
  answer=io.read()

  io.write(say(answer))

until answer=="end"
