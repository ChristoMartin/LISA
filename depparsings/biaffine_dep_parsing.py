from supar import Parser

with open('conll05.plain') as f:
    lines = f.readlines()
    sents = [line.split(' ') for line in lines]

for sent in sents[:3]:
    print(sent)
parser = Parser.load('biaffine-dep-en')
dataset = parser.predict(sents[:5], prob=True, verbose=False)