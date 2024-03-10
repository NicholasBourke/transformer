Basic Transformer architecture based on "Attention Is All You Need" (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin; 2017)

The intention is to gain deeper understanding of the Transformer network architecture by replicating the model from the source (ie. the research paper itself). Some secondary sources were used however (eg. https://jalammar.github.io/illustrated-transformer/). The model was contructed for the purpose of translation, as in the paper, and so may lack the more general applicability that transformers can have.



PROGRESS

- Basic structure seems to be working as expected, but yet to write training functions or work with any actual data. Forward proagation runs without any issues on dummy data in main.py and outputs are as expected.


DISCUSSION:

- In some secondary references I saw it seemed as if encoder-decoder attention was masked much like decoder self-attention. From what I can tell, encoder-decoder attention is not masked in the architecture outlined in the paper. This makes sense as surely we want target positions to be able to attend to all source positions, and so that is how it has been constructed here.