
--Modified by Mike Tanana from Andrew Karpathy and Wojciech Zaremba
--Changed to support word models and Seq2Seq

--input comes in the form of a file where each line has a speaker and response
--in the form:  speakerone utterance | speaker two response



local WordSplitLMMinibatchLoader = {}
WordSplitLMMinibatchLoader.__index = WordSplitLMMinibatchLoader

WordSplitLMMinibatchLoader.tokenizer = require "tokenizer"



function WordSplitLMMinibatchLoader.shuffle(t)
    local n = #t
    while n > 2 do
        local k = math.random(n)
        t[n], t[k] = t[k], t[n]
        n = n - 1
    end
    return t
end

function WordSplitLMMinibatchLoader.createFromJustVocab(vocabfile)
    local self = {}
    setmetatable(self, WordSplitLMMinibatchLoader)
    self:loadExistingVocabFile(vocabfile)

    return self

end

function WordSplitLMMinibatchLoader:loadExistingVocabFile(vocabfilename)

    --this is word to index
    self.vocab_mapping = torch.load(vocabfilename)
    self.id2word = {}
    self.word2id=self.vocab_mapping

    --count vocab and make reverse mapping
    self.vocab_size = 0
    for word,idx in pairs(self.vocab_mapping) do
        self.vocab_size = self.vocab_size + 1
        self.id2word[idx] = word
    end
    print('Vocab Size'..self.vocab_size)



    self.goToken = self.vocab_mapping['<go>']
    self.eosToken = self.vocab_mapping['<eos>']
    self.unknownToken = self.vocab_mapping['<unk>']


end

function WordSplitLMMinibatchLoader.create(data_dir, batch_size,seq_length, split_fractions,vocabsize)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, WordSplitLMMinibatchLoader)

    local input_file = path.join(data_dir, 'input.txt')
    local vocab_file = path.join(data_dir, 'vocabwords.t7')
    local tensor_file = path.join(data_dir, 'datawords.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        self:text_to_tensor(input_file, vocab_file, tensor_file,vocabsize,split_fractions)
    end

    print('loading data files...')
    --in this file rows are dialogue pairs: first half is speaker1 second half is speaker 2
    --the data should always store at least one more than you are going to predict (otherwise the final step will be incorrect)
    local data = torch.load(tensor_file)
    self.train = data.train
    self.val =data.val
    self.test = data.test

    self:loadExistingVocabFile(vocab_file)

    --shuffle rows
    WordSplitLMMinibatchLoader.shuffleTensorRows(self.train)



    -- divide data to train/val and allocate rest to test
    self.ntrain = math.floor(self.train:size(1)/batch_size )-1
    self.nval = math.floor(self.val:size(1)/batch_size)-1
    self.ntest = math.floor(self.test:size(1)/batch_size)-1
    self.batch_size = batch_size
    print ('Val Size: ' .. self.val:size(1))

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0 }



    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()

    --self:writeTxtFile(1,self.ntrain,self.vocab_mapping,"train.txt");
    --self:writeTxtFile(2,self.nval,self.vocab_mapping,"test.txt");





    return self
end

function WordSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function WordSplitLMMinibatchLoader:writeTxtFile(split_index,n,vocab,filename)
    print("Saving data "..filename)
    --get the numerically indexed vocab table
    local ivocab = {}
    for c,i in pairs(vocab) do ivocab[i] = c end


    local file = io.open(filename, "a")


    for key= 1,n  do
        local inx,outx,outy=self:next_batch(split_index,25,25)
        self:writeBatch(file,outx,ivocab)

    end

    file:close()

end
function WordSplitLMMinibatchLoader:writeBatch(file,batch,ivocab)

    for row =1 , batch:size(1) do
        for col=1 ,  batch:size(2) do
            local word = ivocab[batch[row][col]]
            if(word=="<pad>") then break end
            if(word=="<go>") then
                file:write("")
            else
                file:write(word.." ")
            end

        end
        file:write("\n")
    end

end




function WordSplitLMMinibatchLoader:getBatch(split_index,batchid,insize,outsize)
    local set = {}
    if split_index ==1 then set = self.train
    elseif split_index==2 then set = self.val
    elseif split_index==3 then set = self.test
    end



    -- pull out the correct next batch
    local start = (batchid*self.batch_size)+1



    local x = set:narrow(1,start,self.batch_size)
    local y = getydataforx(x)
    --print(x)
    --print(y)
    --io.read()

    --split the two sequences apart
    local length = x:size(2)
    local size = length/2
    local in_start = math.max(1,size-(insize-1))
    local in_usesize = math.min(size,insize)
    local out_usesize = math.min(size,outsize)

    local inputx = x:narrow(2,in_start,in_usesize)
    --print(x)
    --print(in_start.." "..in_usesize)
    --print(inputx)
    local inputy = y:narrow(2,in_start,in_usesize)
    inputx,inputy = self:trimPaddingFromLeft(inputx,inputy)
    local outputx = x:narrow(2,size+1,out_usesize)
    local outputy = y:narrow(2,size+1,out_usesize):clone()
    outputx,outputy = self:trimPaddingFromRight(outputx,outputy)
    --need to do this because the criterion can't handle zeros even though they get masked
    outputy[outputy:lt(1)]=self.eosToken
    return inputx:t(),outputx:t(),outputy:t()
end
--trims to the first non padding elemend
--assumes t1 and t2 are the same dimensions
function WordSplitLMMinibatchLoader:trimPaddingFromLeft(t1,t2)
    local firstValid = 0
    for i = 1, t1:size(2) do
        for j = 1, t1:size(1) do
            local val = t1[j][i]
            local val2 = t2[j][i]
            if(val~=0 or val2~=0) then
                firstValid=i
                break
            end
        end
        if(firstValid > 0) then break end
    end
    if(firstValid==0) then return t1,t2 end
    local length = t1:size(2)+1-firstValid
    local newt1 = t1:narrow(2,firstValid,length)
    local newt2 = t2:narrow(2,firstValid,length)

    return newt1,newt2

end
--trims to there
function WordSplitLMMinibatchLoader:trimPaddingFromRight(t1,t2)
    local firstValid = 0
    for i = t1:size(2), 1,-1 do
        for j = 1, t1:size(1) do
            local val = t1[j][i]
            local val2 = t2[j][i]
            if(val~=0 or val2~=0) then
                firstValid=i
                break
            end
        end
        if(firstValid > 0) then break end
    end
    if(firstValid==0) then return t1,t2 end
    local length = firstValid
    local newt1 = t1:narrow(2,1,firstValid)
    local newt2 = t2:narrow(2,1,firstValid)

    return newt1,newt2
end





function getydataforx(xdata)
    xt = xdata:t()
    yt = xt:clone()  --watch out transpose works off the same data
    ydata = yt:sub(1,-2):copy(xt:sub(2,-1)) --shift everything down one
    yt[-1] = xt[1] --make the last item the same as the first (i.e. make sure you dont' set a seq length that actually uses this)
    y = yt:t()  --put back into cols are seq length and rows are samples
    return y
end


--[[
--
 - deprecated user WordSplitLMMinibatchLoader.tokenizer.tokenize(t)-
function WordSplitLMMinibatchLoader.preprocess(alltext)
  --make sure there are spaces around certain characters so that we predict them as individual units
  local newtext
  newtext = alltext:gsub(',',' , ')
  newtext = newtext:gsub('%.',' . ')
  newtext = newtext:gsub('%:',' : ')
  newtext = newtext:gsub('%;',' ; ')
  newtext = newtext:gsub('%?',' ? ')
  newtext = newtext:gsub('%!',' ! ')
  newtext = newtext:gsub('\n',' \n ')


  return newtext
end]]--

---Makes sure we split on spaces
----return nil if we are at the end
function getNextBatchFromFile(torchfile)
    --first get main buffer size and create a string
    local chars = torchfile:readByte(100000);
    if(chars:size()==0) then return nil end
    local text = chars:string();
    --now make keep going until we get a space (or it is the end)
    local nospace=true;
    local extrachars = "";
    while nospace do
        local char =torchfile:readByte()
        if char==nil or string.char(char)==" " then break end
        extrachars=extrachars..string.char(char)
    end
    text=text..extrachars
    --io.write(text)
    --return text
    return text

end

function  WordSplitLMMinibatchLoader.shuffleTensorRows(t)
    --shuffle tensor
    local indexes = torch.randperm(t:size(1)):type('torch.LongTensor')
    t = t:index(1,indexes)
    return t

end
---Makes sure we split on new lines
----return nil if we are at the end
function WordSplitLMMinibatchLoader.getNextBatchFromFileStandard(file)
    --first get main buffer size and create a string
    local block= file:read(1000000);
    if not block then return nil end
    local text = block;
    --now make keep going until we get a space (or it is the end)
    local nospace=true;
    local extrachars = "";
    while nospace do
        local char =file:read(1)
        if char==nil or char=="\n" or char=="\r" then break end
        extrachars=extrachars..char
    end
    text=text..extrachars
    --io.write(text)
    --return text
    return text

end

--input comes in the form of a file where each line has a speaker and response
--in the form:  speakerone utterance | speaker two response

-- *** STATIC method ***
function WordSplitLMMinibatchLoader:text_to_tensor(in_textfile, out_vocabfile, out_tensorfile,vocabsize,split_fractions)
    --local timer = torch.Timer()
    local matchstring = "([^%s]+)"
    print('loading text file...')
    local wordcount = {}
    local rawdata
    local tot_len = 0
    local filein = io.open(in_textfile, "r")
    --local filein = torch.DiskFile(in_textfile, "r")
    --filein:quiet();
    local unknownword = "<unk>"
    local padding = "<pad>" --pads are now just zeros
    local go = "<go>"
    local eos = "<eos>"

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all characters to a set
    local unordered = {}
    local count=0
    local t=true
    local nlines =0


    while(t ~= nil) do
        t=WordSplitLMMinibatchLoader.getNextBatchFromFileStandard(filein)
        if t ==nil then break end
        -- t=WordSplitLMMinibatchLoader.preprocess(t)

        local words = WordSplitLMMinibatchLoader.tokenizer.tokenize(t)


        for word in words do
            word = word:lower()
            if word ~= "|" then  --speaker change character
            if wordcount[word]==nil then
                wordcount[word]=1
            else
                wordcount[word]=wordcount[word]+1
            end
            tot_len=tot_len+1
            else  --if word== "|"  easy way to count n dialog elements
            nlines=nlines+1
            end
        end
        io.write(tot_len.."\n")

    end


    filein:close()






    --------------------------------------------------------
    --trim vocabulary---------------------------------------
    --------------------------------------------------------

    --basically start at some very high frequency and then go down until we have added the right number of words
    --the ties will kind of be added 'randomly'  (really based on which were put in first)
    local frequency = 400  --start here and go down

    local vocab_mapping = {}
    local index=1
    --add special words
    vocab_mapping[unknownword]=index;
    --index=index+1
    --vocab_mapping[padding]=index;
    index=index+1
    vocab_mapping[go]=index;
    index=index+1
    vocab_mapping[eos]=index;
    index=index+1
    local count=0
    while frequency >0 do
        for key,value in pairs(wordcount) do
            if(value>=frequency) then  --trim dictionary for rare words
            vocab_mapping[key]=index;
            index=index+1
            count=count+1
            wordcount[key]=nil --remove from table
            if(count>=vocabsize) then break end
            end
        end
        if(count>=vocabsize) then break end
        frequency=frequency-1
    end



    print("Count: "..count)
    print("Length: "..tot_len)

    --------------------------------------------------------
    --build dataset---------------------------------------
    --------------------------------------------------------

    local length = 50 --size to save of each utterance

    -- construct a tensor with all the data
    print('putting data into tensor...')

    --fill with pads first
    --rows are dialogue examples by length of examples (*2 b/c utterance-response)
    local examples = torch.IntTensor(nlines,length*2):fill(0)



    filein = io.open(in_textfile, "r")

    t=true
    local row=1
    while(t ~= nil) do
        t=WordSplitLMMinibatchLoader.getNextBatchFromFileStandard(filein)
        if t ==nil then break end
        --break into lines
        local lines = t:gmatch("[^\r\n]+")
        for line in lines do

            --line=WordSplitLMMinibatchLoader.preprocess(line)
            --break into words
            local words = WordSplitLMMinibatchLoader.tokenizer.tokenize(line)

            local speaker1,speaker2 = WordSplitLMMinibatchLoader.getSpeakersForLine(words)


            --fill speaker 1 from middle->start
            --speaker 1 we want to go from the last word spoken backward
            local count=0
            for i =speaker1.size,1,-1 do
                local word = speaker1[i]
                if(count<length) then
                    if(word==nil) then for i, v in ipairs(speaker1) do print(i, v) end end
                    word = word:lower()
                    local idx = vocab_mapping[word]
                    if idx == nil then idx = vocab_mapping[unknownword] end
                    local loc = length-count
                    examples[row][loc]=idx
                end
                count=count+1
            end
            --speaker 2 middle->end
            for i =1,speaker2.size do
                local word = speaker2[i]
                if(i<=length) then
                    word = word:lower()
                    local idx = vocab_mapping[word]
                    if idx == nil then idx = vocab_mapping[unknownword] end
                    local loc = length+i
                    examples[row][loc]=idx
                end
            end
            --debugging here:
            --print(examples[row])
            --io.stdin:read'*l'

            row=row+1

        end


    end

    --splits


    local ntrain = math.floor(examples:size(1) * split_fractions[1])
    local nval = math.floor(examples:size(1) * split_fractions[2])
    local ntest = examples:size(1) - nval - ntrain -- the rest goes to test (to ensure this adds up exactly)

    --shuffle tensor
    local indexes = torch.randperm(examples:size(1)):type('torch.LongTensor')
    examples = examples:index(1,indexes)

    local data = {}
    data.train = examples:narrow(1,1,ntrain)
    data.val = examples:narrow(1,ntrain+1,nval)
    data.test = examples:narrow(1,ntrain+nval+1,ntest)


    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, data)




end
--extract speakers from a single line
function WordSplitLMMinibatchLoader.getSpeakersForLine(words)
    local speaker1 = {}
    speaker1.size=0
    local speaker2 = {}
    speaker2.size=0
    local isspeaker1=true
    for word in words do
        if word=="|" then
            isspeaker1=false
            speaker2[speaker2.size+1]="<go>"
            speaker2.size=speaker2.size+1
        else
            if isspeaker1==true then
                speaker1[speaker1.size+1]=word
                speaker1.size=speaker1.size+1
            else
                speaker2[speaker2.size+1]=word
                speaker2.size=speaker2.size+1
            end
        end
    end

    --add end of speaker tag
    speaker2[speaker2.size+1]="<eos>"
    speaker2.size=speaker2.size+1


    return speaker1,speaker2

end







return WordSplitLMMinibatchLoader