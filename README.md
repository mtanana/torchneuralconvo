# torchneuralconvo

This is really an extension of awesome work from the Element Research People:  (rnn) and macournoyer's great project (neralconvo)

It's completely in beta form now (I'll post more when I get the bugs out)

The nice features added from the above projects are the ability to do 1) Ada-grad 2) gradient clipping 3)  multilayer LSTM's 4) load larger datasets 5) Set explicit vocab size 6)  You can run separate train/test splits

Might add beam search soon...

To run, use th trainoptim.lua --cuda  

(well-  you don't have to use cuda, but this would be crazy...)

I put a small sample from the opensubs dataset up.  Really, you can add any dataset in the form:  

speaker1 |  response

every new line is a new pair.   


