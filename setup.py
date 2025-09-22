from setuptools import setup, find_packages

setup(
    name="hybrid_expert_chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "faiss-cpu",
        "spacy",
        "numpy",
    ],
)