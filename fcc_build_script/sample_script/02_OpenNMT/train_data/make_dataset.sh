wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
tar xf toy-ende.tar.gz
cd toy-ende
onmt-build-vocab --size 50000 --save_vocab src-vocab.txt src-train.txt
onmt-build-vocab --size 50000 --save_vocab tgt-vocab.txt tgt-train.txt
