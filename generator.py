from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self):
        # Load tokenizer and model (lightweight CPU-compatible)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(self, context, query):
        # Refined prompt template
        prompt = (
            "You are a helpful assistant. Use only the information provided in the context below to answer the user's question. "
            "If the answer cannot be found in the context, respond with 'The answer is not available in the provided document.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        # Tokenize input with truncation to fit model limits
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        # Generate output with deterministic decoding
        output = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,      # deterministic: disables randomness
            temperature=0.0       # controls creativity: 0 = most factual
        )

        # Decode and return the output string
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


