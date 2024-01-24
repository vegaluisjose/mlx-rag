# MLX RAG

Explore a simple example of utilizing [MLX](https://github.com/ml-explore/mlx) for RAG application running locally on your Apple Silicon device.

I have previously converted the weights for the embedding model [gte-large](https://huggingface.co/thenlper/gte-large) into MLX format, and you can find them stored [here](https://huggingface.co/vegaluisjose/mlx-rag) in the mlx-rag repository. Additionally, as a base model, I am using [NeuralBeagle14-7B-4bit-mlx](https://huggingface.co/mlx-community/NeuralBeagle14-7B-4bit-mlx).



## Getting started

* Install requirements
```bash
python3 -m pip install -r requirements.txt
```

* Create vector database from a pdf file
```bash
python3 create_vdb.py --pdf flash_attention.pdf --vdb vdb.npz
```

* Query database (pdf file)
```bash
python3 query_vdb.py --question "what is flash attention?"
```
