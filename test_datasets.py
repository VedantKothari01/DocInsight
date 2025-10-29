from datasets import load_dataset

conll = load_dataset("wikiann", 'en', split="train[:5000]", trust_remote_code=True)
print(conll)