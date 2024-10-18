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

# Cargar el archivo JSON con las preguntas y respuestas procesadas
with open("preguntas_respuestas_procesadasV1.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Función para realizar la búsqueda en el índice FAISS
def search(query, k=5):
    # Generar el embedding de la consulta
    query_embedding = model.encode(query).astype('float32')

    # Normalizar el embedding
    query_embedding /= np.linalg.norm(query_embedding)

    # Realizar la búsqueda en FAISS
    D, I = index.search(np.array([query_embedding]), k * 2)  # Buscar más resultados por si hay muchos de tipo "info"

    results_qa = []  # Para almacenar resultados de tipo "qa"
    results_info = []  # Para almacenar resultados de tipo "info"

    for i in range(len(I[0])):
        # Obtener el ID del resultado (que corresponde al índice en tu dataset)
        idx = int(I[0][i])

        # Obtener el contenido asociado al índice (de tu dataset)
        result_data = data[idx]

        # Verificar el tipo de contenido y clasificarlo
        if result_data['type'] == 'qa':
            results_qa.append({
                'pregunta': result_data['content'].get('pregunta', ''),
                'respuesta': result_data['content'].get('respuesta', ''),
                'url': result_data.get('url', ''),
                'tipo': result_data.get('type', ''),
                'metadata': result_data.get('metadata', {}),
                'similarity_score': float(D[0][i])  # Puntuación de similitud
            })
        elif result_data['type'] == 'info':
            results_info.append({
                'titulo': result_data['content'].get('titulo', ''),
                'descripcion': result_data['content'].get('descripcion', ''),
                'url': result_data.get('url', ''),
                'tipo': result_data.get('type', ''),
                'metadata': result_data.get('metadata', {}),
                'similarity_score': float(D[0][i])  # Puntuación de similitud
            })

        # Si ya tenemos suficientes resultados de tipo "qa", salimos del bucle
        if len(results_qa) >= k:
            break

    # Si no hay suficientes resultados "qa", complementamos con los de tipo "info"
    if len(results_qa) < k:
        results_qa.extend(results_info[:k - len(results_qa)])  # Añadir los primeros resultados de "info" si es necesario

    return results_qa

# Vista para manejar la búsqueda en el índice FAISS
@csrf_exempt
def search_view(request):
    if request.method == 'POST':
        # Verificar si el cuerpo de la solicitud está vacío
        if not request.body:
            return JsonResponse({'error': 'Empty request body'}, status=400)

        # Intentar decodificar el cuerpo JSON
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

        query = data.get('query')
        if not query:
            return JsonResponse({'error': 'No query provided'}, status=400)

        # Realizar la búsqueda y devolver los resultados
        results = search(query)
        return JsonResponse(results, safe=False)
    else:
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
