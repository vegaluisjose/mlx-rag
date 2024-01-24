import argparse
from vdb import VectorDB
from mlx_lm import load, generate

TEMPLATE = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

CONTEXT:

{context}

Question: {question}
Answer:
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a vector DB")
    # Input
    parser.add_argument(
        "--question",
        help="The question that needs to be answered",
        default="what is flash attention?",
    )
    # Input
    parser.add_argument(
        "--vdb",
        type=str,
        default="vdb.npz",
        help="The path to read the vector DB",
    )
    args = parser.parse_args()
    m = VectorDB(args.vdb)
    context = m.query(args.question)
    prompt = TEMPLATE.format(context=context, question=args.question)
    model, tokenizer = load("mlx-community/NeuralBeagle14-7B-4bit-mlx")
    generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=512)
