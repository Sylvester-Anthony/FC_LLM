import re
import tiktoken


with open("/Users/sylvesteranthony/Documents/FC_LLM/data/datacommentary.txt","r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


# all_words = sorted(set(preprocessed))

# vocab = {token:integer for integer, token in enumerate(all_words)}

all_tokens = sorted(list(set(preprocessed)))

all_tokens.extend(["<|endoftext|>","<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

len(vocab)


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self,text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self,text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizerv1 = SimpleTokenizerV1(vocab)
tokenizerv2 = SimpleTokenizerV2(vocab)
text = """
         GOAL!!! HAALAND PUTS CITY AHEAD!!! Having netted 36 Premier League goals last season, it has taken just four minutes for him to open his account for this campaign. It is a typical Haaland goal as he lurks in the box while De Bruyne lofts a deep cross for Rodri. The midfielder nods it back across, and the Norway international is waiting to smash home and put the champions 1-0 up!
Rodri's header was clever to put it on a plate for Haaland, who finished with typical aplomb.
       """
ids = tokenizerv1.encode(text)
print(ids)
print(tokenizerv1.decode(ids))

text_2 = "Hello do you like tea ?"
print(tokenizerv2.encode(text_2))
print(tokenizerv2.decode(tokenizerv2.encode(text_2)))



text3 = "Hello, do you like tea?"
text4 = "In the sunlit terraces of the palace."
joined_text = " <|endoftext|> ".join((text3, text4))
print(tokenizerv2.encode(joined_text))
print(tokenizerv2.decode(tokenizerv2.encode(joined_text)))


tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
        
strings = tokenizer.decode(integers)
print(strings)


enc_text = tokenizer.encode(raw_text)
all_tokens = sorted(list(set(enc_text)))
print(f"the vocab size is:",len(all_tokens))

enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x})") 
print(f"y:     {y}")


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "------>", desired)
    
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "------>", tokenizer.decode([desired]))
