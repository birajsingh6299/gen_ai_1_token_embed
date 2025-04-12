import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4o-mini')

print("Vocab size", encoder.n_vocab)

text = "The cat sat on a mat"
tokens = encoder.encode(text)

print("Tokens", tokens)

decoded = encoder.decode(tokens)
print("Decoded", decoded)
