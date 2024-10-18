from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Cargar el modelo de Hugging Face (mismo que usamos para generar los embeddings)
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Cargar el índice FAISS y los embeddings guardados
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.read_index("faiss_index.index")

# Función para realizar la búsqueda en el índice FAISS
def search(query, k=3):
    # Generar el embedding de la consulta
    query_embedding = model.encode(query).astype('float32')

    # Normalizar el embedding
    query_embedding /= np.linalg.norm(query_embedding)

    # Realizar la búsqueda en FAISS
    D, I = index.search(np.array([query_embedding]), k)

    results = []
    for i in range(k):
        result = {
            'id': int(I[0][i]),  # ID del documento
            'similarity_score': float(D[0][i]),  # Puntuación de similitud
        }
        results.append(result)

    return results

# Vista para manejar la búsqueda en el índice FAISS
@csrf_exempt  # Desactivar la protección CSRF para peticiones POST
def search_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query = data.get('query')

        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)

        # Realizar la búsqueda y devolver los resultados
        results = search(query)
        return JsonResponse(results, safe=False)
    else:
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
