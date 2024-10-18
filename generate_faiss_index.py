import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Cargar el modelo de Hugging Face
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Cargar tus datos desde el archivo JSON
with open("preguntas_respuestas_procesadasV1.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Procesar y generar embeddings
processed_data = []
embeddings = []

for item in data:
    if item['type'] == 'qa':
        text_for_embedding = f"{item['content']['pregunta']} {item['content']['respuesta']}"
    elif item['type'] == 'info':
        text_for_embedding = f"{item['content']['titulo']} {item['content'].get('descripcion', '')}"

    # Generar embedding con el modelo
    embedding = model.encode(text_for_embedding)

    # Guardar embedding en lista
    embeddings.append(embedding)
    processed_data.append({
        'text_for_embedding': text_for_embedding,
        'full_content': item['content'],
        'type': item['type'],
        'url': item.get('url', ''),
        'metadata': item.get('metadata', {})
    })

# Guardar embeddings en archivo .pkl
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# Crear el índice FAISS
embedding_matrix = np.array(embeddings).astype('float32')

# Normalizar los embeddings para que sean adecuados para búsquedas por similitud
embedding_matrix /= np.linalg.norm(embedding_matrix, axis=1)[:, None]

# Crear el índice FAISS usando producto punto
index = faiss.IndexFlatIP(embedding_matrix.shape[1])

# Añadir los embeddings al índice FAISS
index.add(embedding_matrix)

# Guardar el índice FAISS
faiss.write_index(index, "faiss_index.index")

print("Embeddings y FAISS index generados correctamente")
