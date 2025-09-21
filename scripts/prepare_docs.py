import os
import pathlib

def clean(text):
    return " ".join(text.split())

out = []
for p in pathlib.Path("docs").glob("*"):
    text = p.read_text(encoding="utf-8")
    (p.with_suffix('.clean.txt')).write_text(clean(text), encoding="utf-8")
