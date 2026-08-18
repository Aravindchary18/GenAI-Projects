from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(texts):
   
    return model.encode(
        
        texts,
        batch_size=32,
        show_progress_bar=False

    )