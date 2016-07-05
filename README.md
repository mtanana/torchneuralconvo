## Torch Neural Conversation Model

This is an implementation of seq2seq for language models implemented in torch.

The main features are:
1.  Multilayer LSTM's
2.  Batch Processing
3.  Explicit Vocabulary Size
4.  Adagrad (but easy to use any torch.optim plugins)
5.  Train/Test split
6.  Gradient clipping
7.  Large dataset support (above the normal LUA JIT limits, but limited by your system RAM)



This is really an extension of awesome work from the Element Research People:  (rnn) and macournoyer's great project (neralconvo) and some helpful code from karpathy's char-rnn.

This is technically in beta form, but I have confirmed that it is working.



Might add beam search soon...

##Installation

## Tests

To run, use th trainoptim.lua --cuda  

(well-  you don't have to use cuda, but this would be crazy...)

##Dataset

I put a small sample from the opensubs dataset up.  Really, you can add any dataset in the form:  

input |  response

with the pipe ('|') dividing the two.  You should preprocess your data a bit if you use it like this.  (Lua isn't the greatest for writing this kind pf preprocessing)

every new line is a new pair.   


