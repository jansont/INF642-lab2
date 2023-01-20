# INF642-lab2

The idea is to develop a sequence-to-sequence model using Transformers [1]. This method is a machine translation method that maps an input sequence to an output sequence.
The core idea behind the Transformer model is self-attention—the ability to attend to different positions of the input sequence to compute a representation of that sequence. The Transformer model handles variable-sized input using stacks of self-attention layers.
After training the model, you will be able to input a Fundamental Frequency (F0) sequence and return 6 sequences of Action Units: AU01, AU02, AU04, AU05, AU06 and AU07.
