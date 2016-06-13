--
-- Created by IntelliJ IDEA.
-- User: user
-- Date: 3/13/2016
-- Time: 2:55 PM
-- To change this template use File | Settings | File Templates.
--

local tokenizer = require "tokenizer"
local list = require "pl.List"


-- Word IDs to sentence
function pred2sent(wordIds,dataset)
    local words = {}


    for _, wordId in ipairs(wordIds) do
        local id = wordId
        if id ~= 0 and id~=dataset.goToken and id~=dataset.eosToken then
            local word = dataset.id2word[id]
            table.insert(words, word)
        end
    end
    --print(words)
    return tokenizer.join(words)
end

function tensor2sent(wordIds, dataset)
    local words = {}


    for i=1,wordIds:size(1) do
        local id = wordIds[i][1]
        if id ~= 0 then
            local word = dataset.id2word[id]
            table.insert(words, word)
        end
    end

    return tokenizer.join(words)
end

function printmytable(t)
    for i,v in ipairs(t) do
            print(v)
    end

end

--word ids and probabilites are both tables of the length of the final output

function printProbabilityTable(wordIds, probabilities, num)
    print(string.rep("-", num * 22))
   -- printmytable(wordIds)
   -- printmytable(probabilities)
    --p is the final output word id
    for p, wordId in ipairs(wordIds) do

        local line = "| "
        local probs = probabilities[p];
        for i = 1, num do
           local pr =  probs[1][i]
           local w = wordId[1][i]
           local word = dataset.id2word[w]
          -- print(word)
         --  local t = probabilities[1][p]
         --  print("prob.."..t)
         -- print("wordid.."..wordId[1][i])
          line = line .. string.format("%-10s(%4d%%)", word, pr * 100) .. "  |  "
        end
        print(line)
    end

    print(string.rep("-", num * 22))
end

function getResponse(text,dataset,model,debug)
    debug = debug or false
    local wordIds = {}

    for word in tokenizer.tokenize(text) do
        local id = dataset.word2id[word] or dataset.unknownToken
        table.insert(wordIds, id)
    end

    local input = torch.Tensor({wordIds}):t()
    --print(input)
    local output,predictions, probabilities = model:eval(input)
   -- print(output)
    local phrase = pred2sent(output,dataset)

    if debug then
       printProbabilityTable(wordIds, probabilities, 4)
    end
    phrase = phrase or ''
    return phrase

end



