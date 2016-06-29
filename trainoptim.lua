require 'neuralconvo'
require 'xlua'
require 'util.ModelTracker'
require 'nn'
require 'util.Tester'

local WordSplitLMMinibatchLoader = require 'util.WordSplitLMMinibatchLoader'

torch.setheaptracking(true)

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--dataDir', 'data/opensubssmall/', 'approximate size of dataset to use (0 = all)')
cmd:option('--vocabSize', 15000, 'Vocab Size')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--hiddenSize', 1000, 'number of hidden units in LSTM')
cmd:option('--nlayers', 2, 'Number of Layers')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
--cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 20, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxEpoch', 10, 'maximum number of epochs to run')
cmd:option('--batchSize', 1, 'minibatch size')
cmd:option('--seqLength',50,'Max Sequence Length');
cmd:option('-seq_length_in',25,'length of sequence input')
cmd:option('-seq_length_out',25,'length of sequence output')


--Mike Additions

cmd:option('--grad_clip',5,'clip gradients at this value ')
cmd:option('--track',0,'Use ModelTracker')
cmd:option('--supermodelid',30627892,'Modeltracking- Supermodel ID')
cmd:option('--rmsprop', false, 'use RMSProp')
cmd:text()
options = cmd:parse(arg)

if options.dataset == 0 then
    options.dataset = nil
end


--for modeltracking online
local crossid=-99
if(options.track==1) then
    local desc = ""
    for k, v in pairs( options ) do desc = desc..k..": "..tostring(v).." " end

    local sm=ModelTracker.createSubmodel({["name"]="Neuraltalk lr:"..options.learningRate.." ",["description"]=desc,["supermodelid"]=options.supermodelid})
    local cross= ModelTracker.createCross({["name"]="Main",["description"]="Main Cross",["submodelid"]=sm.submodelid})
    crossid=cross.crossid
end





-- Data
print("-- Loading dataset")
--[[
dataset = neuralconvo.DataSet(neuralconvo.OpensubsDialogs("data/opensubs"),
    {
        loadFirst = options.dataset,
        minWordFreq = options.minWordFreq
    })
]]--
local loader = WordSplitLMMinibatchLoader.create(options.dataDir, options.batchSize, options.seqLength, {.945, .0001, .05} ,options.vocabSize)


-- Model
model = neuralconvo.Seq2Seq(loader.vocab_size, options.hiddenSize,options.grad_clip,options.nlayers)
model.goToken = loader.goToken
model.eosToken = loader.eosToken
model.unknownToken = loader.unknownToken

-- Training parameters
model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
model.learningRate = options.learningRate
--model.momentum = options.momentum
local decayFactor = (options.minLR - options.learningRate) / options.saturateEpoch
local minMeanError = nil

model.optimState.learningRate=options.learningRate

print('Loading Model')
-- Enabled CUDA
if options.cuda then
    require 'cutorch'
    require 'cunn'
    model:cuda()
end







function runValidationSet()
    local n = loader.nval
    local splitIndex = 2
    local testerr= torch.Tensor(n):fill(0)
    for i = 1,n do

        local inputx,outputx,outputy=loader:getBatch(splitIndex ,i,options.seq_length_in,options.seq_length_out)
        if(inputx:nDimension()~=0 and outputx:nDimension()~=0 and outputy:nDimension()~=0  and outputx:nonzero():nDimension()~=0) then
            if options.cuda then
                inputx = inputx:cuda()
                outputx = outputx:cuda()
                outputy = outputy:cuda()

            end

            local minibatch={}
            table.insert(minibatch,{input=inputx,target=outputx,targetout=outputy})
            local err = model:train(inputx, outputx,outputy,false)

            testerr[i]=err
        end
        xlua.progress(i, n)
    end

    print("Validation Error: "..testerr:mean())
    collectgarbage()
    return testerr:mean()


end

local reportEvery = 1000
local testEvery=20000
local first=true

-- Run the experiment
local totalcount=1
for epoch = 1, options.maxEpoch do
    print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch)
    print("")

    --shuffle training batches
    loader.train =  WordSplitLMMinibatchLoader.shuffleTensorRows(loader.train)


    local errorssmall = torch.Tensor(reportEvery):fill(0)
    local timer = torch.Timer()


    local j= 1

    for i = 1, loader.ntrain do
        if(1%200==0) then collectgarbage() end
        local inputx,outputx,outputy  =loader:getBatch(1 ,i,options.seq_length_in,options.seq_length_out)

        local encInSeq = inputx;
        local decInSeq= outputx;
        local decOutSeq=outputy;

        --[[
        print('EncIn')
        print(encInSeq)
        print(tensor2sent(encInSeq,loader))
        print('DecIn')
        print(tensor2sent(decInSeq,loader))
        print('DecOut')
        print(tensor2sent(decOutSeq,loader))

        io.read()
        ]]--

        if(inputx:nDimension()~=0 and outputx:nDimension()~=0 and outputy:nDimension()~=0 and outputx:nonzero():nDimension()~=0 ) then
            --have to do this before cuda'ing
            local nonzerot = decInSeq:nonzero():size(1)
            if options.cuda then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
                encInSeq = encInSeq:float():cuda()
                decInSeq = decInSeq:float():cuda()
                decOutSeq = decOutSeq:float():cuda()
            end
            local minibatch={}
            table.insert(minibatch,{input=encInSeq,target=decInSeq,targetout=decOutSeq,nonzeroTargets=nonzerot})


            local err
            if options.rmsprop then
                err= model:trainOptim(minibatch, "rmsprop")
                if(first) then
                    print("Using RMSProp")
                    first=false
                end

            else err= model:trainOptim(minibatch, "adagrad")
            end

            errorssmall[j]=err
            j=j+1
        end
        if j == reportEvery then

            print(string.format("Error = %.3f", (errorssmall:mean()) )..string.format(" Progress = %.1f", (totalcount)) )
            if(options.track==1) then
                pcall(ModelTracker.sendStatistic({["category"]="Next",["name"]="Loss",["group"]="train",["n"]=totalcount,["crossid"]=crossid,["value"]=errorssmall:mean()}))
            end
            errorssmall = torch.Tensor(reportEvery):fill(0)
            j=1
        end

        xlua.progress(i, loader.ntrain)
        i = i + 1
        --test set
        if i % testEvery==0 then
            local meanerr=runValidationSet()
            if(options.track==1) then
                pcall(ModelTracker.sendStatistic({["category"]="Next",["name"]="Loss",["group"]="test",["n"]=totalcount,["crossid"]=crossid,["value"]=meanerr}))
            end
            print("Hi : ".. getResponse("Hi",loader,model))
            print("What is your name : ".. getResponse("What is your name",loader,model))
            print("How old are you : ".. getResponse("How old are you ",loader,model))
            print("What is the meaning of life : ".. getResponse("What is the meaning of life ",loader,model))

        end

        if i% 1000 ==0 then
            print("Hi : ".. getResponse("Hi",loader,model))
            print("What is your name : ".. getResponse("What is your name",loader,model))
            print("How old are you : ".. getResponse("How old are you ",loader,model))

        end


        if(totalcount % 1000000==0) then
            print("\n(Saving model ...)")
            torch.save("data/model.t7", model)

        end

        if(totalcount%100000==0 and options.track==1)then


            local report=""
            report=report.."<p>Hi : "..getResponse("Hi",loader,model).."<p>"
            report=report.."<p>What is your name  : "..getResponse("What is your name ",loader,model).."<p>"
            report=report.."<p>How old are you : "..getResponse("How old are you ",loader,model).."<p>"
            report=report.."<p>What is the meaning of life : "..getResponse("What is the meaning of life",loader,model).."<p>"
            report=report.."<p>Do you like swimming : "..getResponse("Do you like swimming ",loader,model).."<p>"
            report=report.."<p>It's been a long day : "..getResponse("It's been a long day ",loader,model).."<p>"
            report=report.."<p>goodbye : "..getResponse("goodbye ",loader,model).."<p>"

            ModelTracker.sendReport({["reportname"]="Dialogue At Epoch: "..epoch.." Iteration: "..i,["parentid"]=crossid,["report"]=report})


        end





        totalcount=totalcount+1


    end

    timer:stop()


    print("\nEpoch stats:")

    -- Save the model if it improved.
    --if minMeanError == nil or errors:mean() < minMeanError then
    --    print("\n(Saving model ...)")
    --    torch.save("data/model.t7", model)
    --    minMeanError = errors:mean()
    --end
    print("\n(Saving model ...)")
    torch.save("data/model.t7", model)


    model.learningRate = model.learningRate + decayFactor
    model.learningRate = math.max(options.minLR, model.learningRate)
end


-- Load testing script
require "eval"