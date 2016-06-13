-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("neuralconvo.Seq2Seq")

function Seq2Seq:__init(vocabSize, hiddenSize,clipping,nlayers)
    require 'optim'
    self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
    self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")
    self.useSecondLayer = usesecondlayer or false
    print("Vocab Size: "..vocabSize)
    self.numLayers = nlayers or 1
    print ("Nlayers: ".. self.numLayers)
    self.useSeqLSTM = true -- faster implementation of LSTM + Sequencer

    self:buildModel()

    self.gradientclipping = clipping
end

function Seq2Seq:buildModel()
    -- Encoder
    self.encoder = nn.Sequential()
    self.encoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
    self.encoder.lstmLayers = {}
    for i=1,self.numLayers do
        if self.useSeqLSTM then
            self.encoder.lstmLayers[i] = nn.SeqLSTM(self.hiddenSize, self.hiddenSize)
            self.encoder.lstmLayers[i]:maskZero()
            self.encoder:add(self.encoder.lstmLayers[i])
        else
            self.encoder.lstmLayers[i] = nn.LSTM(self.hiddenSize, self.hiddenSize):maskZero(1)
            self.encoder:add(nn.Sequencer(self.encoder.lstmLayers[i]))
        end
    end
    self.encoder:add(nn.Select(1, -1))

    -- Decoder
    self.decoder = nn.Sequential()
    self.decoder:add(nn.LookupTableMaskZero(self.vocabSize, self.hiddenSize))
    self.decoder.lstmLayers = {}
    for i=1,self.numLayers do
        if self.useSeqLSTM then
            self.decoder.lstmLayers[i] = nn.SeqLSTM(self.hiddenSize, self.hiddenSize)
            self.decoder.lstmLayers[i]:maskZero()
            self.decoder:add(self.decoder.lstmLayers[i])
        else
            self.decoder.lstmLayers[i] = nn.LSTM(self.hiddenSize, self.hiddenSize):maskZero(1)
            self.decoder:add(nn.Sequencer(self.decoder.lstmLayers[i]))
        end
    end
    self.decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(self.hiddenSize, self.vocabSize), 1)))
    self.decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))

    self.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))





    self.encoder:zeroGradParameters()
    self.decoder:zeroGradParameters()






    self.c=nn.Container()
    self.c:add(self.encoder)
    self.c:add(self.decoder)
    self.x,self.dl_dx = self.c:getParameters()
    self.optimState={}

end


function Seq2Seq:cuda()
    self.encoder:cuda()
    self.decoder:cuda()

    if self.criterion then
        self.criterion:cuda()
    end



    self.c:cuda();
    self.x,self.dl_dx = self.c:getParameters()

end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function Seq2Seq:forwardConnect(enc, dec, seqLen)
    for i=1,#enc.lstmLayers do
        if self.useSeqLSTM then
            dec.lstmLayers[i].userPrevOutput = enc.lstmLayers[i].output[seqLen]
            dec.lstmLayers[i].userPrevCell = enc.lstmLayers[i].cell[seqLen]
        else
            dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[seqLen])
            dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[seqLen])
        end
    end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect(enc, dec)
    for i=1,#enc.lstmLayers do
        if self.useSeqLSTM then
            enc.lstmLayers[i].userNextGradCell = dec.lstmLayers[i].userGradPrevCell
            enc.lstmLayers[i].gradPrevOutput = dec.lstmLayers[i].userGradPrevOutput
        else
            enc.lstmLayers[i].userNextGradCell = nn.rnn.recursiveCopy(enc.lstmLayers[i].userNextGradCell, dec.lstmLayers[i].userGradPrevCell)
            enc.lstmLayers[i].gradPrevOutput = nn.rnn.recursiveCopy(enc.lstmLayers[i].gradPrevOutput, dec.lstmLayers[i].userGradPrevOutput)
        end
    end
end











function Seq2Seq:train(input, target,targety, learn)
    --these are just 1d vectors with word ids
    local encoderInput = input
    local decoderInput = target
    local decoderTarget = targety

    if learn == nil then learn =true end

    -- Forward pass
    self.encoder:forward(encoderInput)
    self:forwardConnect(self.encoder,self.decoder,encoderInput:size(1))
    local decoderOutput = self.decoder:forward(decoderInput)
    local Edecoder = self.criterion:forward(decoderOutput, decoderTarget)

    if Edecoder ~= Edecoder then -- Exit early on bad error
    return Edecoder
    end


    -- Backward pass
    if learn then
        local gEdec = self.criterion:backward(decoderOutput, decoderTarget)
        self.decoder:backward(decoderInput, gEdec)
        self:backwardConnect(self.encoder,self.decoder)
        self.encoder:backward(encoderInput, self.zeroTensor)

    end



    self.decoder:forget()
    self.encoder:forget()

    return Edecoder/decoderTarget:size(1)
end

function Seq2Seq:update()

    self.dl_dx:clamp(-self.gradientclipping,self.gradientclipping);

    self.encoder:updateGradParameters(self.momentum)
    self.decoder:updateGradParameters(self.momentum)
    self.decoder:updateParameters(self.learningRate)
    self.encoder:updateParameters(self.learningRate)

    self.encoder:zeroGradParameters()
    self.decoder:zeroGradParameters()


    --self.decoder:forget()
    --self.encoder:forget()

end


function Seq2Seq:trainOptim(minibatch,optimizer)
    --these are just 1d vectors with word ids


    local myseq = self
    local Edecoder;
    local err;

    optimizer=optimizer or "adagrad"

    local feval = function(x_new)
        local totalerr = 0
        local totaln = 0

        for _,example in ipairs(minibatch) do

            --[Note: added a fix from a versioning problem on rnn:
            -- -- https://github.com/macournoyer/neuralconvo/issues/17]--

            local encoderInput = example["input"]
            local target = example["target"]
            local decoderInput = target
            local decoderTarget = example["targetout"]

            local encoderOutput = myseq.encoder:forward(encoderInput)
            myseq:forwardConnect(myseq.encoder,myseq.decoder,encoderInput:size(1))
            local decoderOutput = myseq.decoder:forward(decoderInput)
            -- print(decoderOutput )
            -- print(decoderTarget)
            --io.read()

            local thiserr=myseq.criterion:forward(decoderOutput, decoderTarget)
            local nonzeroinputs = example["nonzeroTargets"]


            totalerr = totalerr+thiserr
            totaln=totaln+decoderTarget:size(1)



            local gEdec = myseq.criterion:backward(decoderOutput, decoderTarget)
            myseq.decoder:backward(decoderInput, gEdec)

            myseq:backwardConnect(myseq.encoder,myseq.decoder)

            myseq.encoder:backward(encoderInput,  torch.Tensor(encoderOutput:size()):zero())


            myseq.decoder:forget()
            myseq.encoder:forget()

        end
        if totaln==0 then err=0
        else err=totalerr/totaln
        end
        myseq.dl_dx:clamp(-self.gradientclipping,self.gradientclipping);
        return err, myseq.dl_dx

    end



    if(optimizer=="adagrad") then _,err=optim.adagrad(feval,self.x ,self.optimState)
    elseif (optimizer=="rmsprop") then _,err=optim.rmsprop(feval,self.x ,self.optimState)
    end
    self.encoder:zeroGradParameters()
    self.decoder:zeroGradParameters()


    self.decoder:forget()
    self.encoder:forget()

    return err[1]
end



















local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input)
    assert(self.goToken, "No goToken specified")
    assert(self.eosToken, "No eosToken specified")


    self.encoder:forward(input)


    local predictions = {}
    local probabilities = {}

    -- Forward <go> and all of it's output recursively back to the decoder
    local output = {self.goToken}

    for i = 1, MAX_OUTPUT_SIZE do
        --wondering if we really need to forward connect before each run because we are
        --kind of starting over each run here
        self:forwardConnect(self.encoder,self.decoder,input:size(1))
        local prediction = self.decoder:forward(torch.Tensor({output}):t())[#output]
        --print(prediction)
        -- prediction contains the probabilities for each word IDs.
        -- The index of the probability is the word ID.
        --2 is to sort over the second dimension
        local prob, wordIds = prediction:topk(5, 2, true, true)

        -- First one is the most likely.
        next_output = wordIds[1][1]
        --use second guess if unk token
        if next_output==self.unknownToken  then next_output = wordIds[1][2] end
        --print(wordIds)
        --print(next_output)
        --io.read()
        table.insert(output, next_output)

        -- Terminate on EOS token
        if next_output == self.eosToken then
            break
        end

        table.insert(predictions, wordIds)
        table.insert(probabilities, prob)
    end

    self.decoder:forget()
    self.encoder:forget()
    self.encoder:zeroGradParameters()
    self.decoder:zeroGradParameters()
    self.decoder:training()
    self.encoder:training()

    return output,predictions, probabilities
end
