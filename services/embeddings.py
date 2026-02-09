from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from typing import List, Dict, Any
import json
from config import config

class EmbeddingService:
    """
    Servicio que convierte productos a vectores y los busca
    
    ¿Qué hace?
    1. Toma un producto y lo convierte a números (embedding)
    2. Guarda esos números en ChromaDB
    3. Cuando buscas, convierte tu búsqueda a números
    4. Encuentra productos con números similares
    """

    def __init__(self):
        # Cargar modelo de embeddings (se descarga la primera vez)
        print("📥 Cargando modelo de embeddings...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Inicializar ChromaDB con persistencia
        print(f"💾 Inicializando ChromaDB en: {config.CHROMA_PATH}")
        self.client = PersistentClient(
            path=config.CHROMA_PATH
        )
        
        # Crear/obtener colección
        self.collection = self.client.get_or_create_collection(
            name="products",
            metadata={"description": "Catálogo de periféricos"}
        )
        
        print("✅ Servicio de embeddings listo")

    def _product_to_text(self, product: Dict[str, Any]) -> str:
        """
        Convierte un producto a texto para crear embedding
        
        Ejemplo:
        Input: {"name": "Mouse Logitech", "brand": "Logitech", ...}
        Output: "Mouse Logitech marca Logitech gaming DPI 16000..."
        """
        parts = [
            product['name'],
            f"marca {product['brand']}",
            product.get('description', ''),
            f"categoría {product['category']}",
            f"subcategoría {product['subcategory']}",
        ]
        
        # Agregar especificaciones
        for spec in product.get('specifications', []):
            parts.append(f"{spec['name']} {spec['value']}")
        
        # Agregar características de variantes
        for variant in product.get('variants', []):
            for feature in variant.get('features', []):
                parts.append(f"{feature['option']} {feature['value']}")
        
        return " ".join(filter(None, parts))
    
    def index_products(self, products: List[Dict[str, Any]]):
        """
        Indexa productos en la base vectorial
        
        ¿Cuándo se usa? Cuando sincronizas el catálogo de Laravel
        """
        print(f"📊 Indexando {len(products)} productos...")
        
        # Eliminar y recrear colección (reset total)
        try:
            self.client.delete_collection("products")
        except Exception:
            pass  # si no existe, no pasa nada

        self.collection = self.client.create_collection(
            name="products",
            metadata={"description": "Catálogo de periféricos"}
        )
        
        texts = []
        metadatas = []
        ids = []
        
        for product in products:
            # Convertir producto a texto
            text = self._product_to_text(product)
            texts.append(text)
            
            # Guardar metadata (info del producto)
            metadatas.append({
                "id": str(product['id']),
                "name": product['name'],
                "brand": product['brand'],
                "category": product['category'],
                "data": json.dumps(product, ensure_ascii=False)
            })
            
            ids.append(f"product_{product['id']}")
        
        # Crear embeddings y guardar
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ {len(products)} productos indexados")
    
    def search(self, query: str, n_results: int = 16) -> List[Dict[str, Any]]:
        """
        Busca productos por similitud semántica
        
        Ejemplo:
        Input: "mouse inalámbrico gaming"
        Output: [productos más similares con score]
        """
        print(f"🔍 Buscando: '{query}'")
        
        # Buscar en ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Procesar resultados
        products = []
        for i, metadata in enumerate(results['metadatas'][0]):
            product_data = json.loads(metadata['data'])
            
            # Calcular porcentaje de similitud (1.0 = 100%)
            # ChromaDB retorna "distance", lo convertimos a similitud
            distance = results['distances'][0][i]
            similarity = max(0, 1 - distance)  # Convertir distancia a similitud
            
            products.append({
                **product_data,
                'similarity_score': round(similarity * 100, 2)  # Porcentaje
            })
        
        return products
    
# Instancia global
embedding_service = EmbeddingService()