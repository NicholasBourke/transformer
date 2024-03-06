


DISCUSSION:

- Decoder self-attention is masked (as we only want target positions attending to earlier positions), however in contradiction to some secondary reference material I ahve seen, encoder-decoder attention is not masked. This is how I believe the paper intends, and also what makes sense to me, as surely we want target positions to attend to all source positions.