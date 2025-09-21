import pathlib
import nltk


from nltk.tokenize import sent_tokenize   # pip install nltk
import json

def chunk_text(text, max_sent_per_chunk=6):
    sents = sent_tokenize(text)
    chunks=[]
    for i in range(0, len(sents), max_sent_per_chunk):
        chunks.append(" ".join(sents[i:i+max_sent_per_chunk]))
    return chunks

all_chunks=[]
for p in pathlib.Path("docs").glob("*.clean.txt"):
    text = p.read_text()
    for c in chunk_text(text):
        all_chunks.append({"doc": p.name, "text": c})

pathlib.Path("data/chunks.json").write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2))
