from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import UserQuery, ConversationResponse
from services.conversation import conversation_service
from services.embeddings import embedding_service
from config import config
import httpx
from typing import List, Dict, Any
import uuid

app = FastAPI(
    title="IA Recomendación Periféricos",
    description="Sistema de recomendación inteligente para periféricos de computadora",
    version="1.0.0"
)

# CORS (permitir que Vue/Mobile se conecten)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Almacenar conversaciones en memoria (en producción usa Redis)
conversations: Dict[str, List[str]] = {}

@app.get("/")
async def root():
    """Endpoint de prueba"""
    return {
        "message": "IA Recomendación Periféricos - API Activa",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/recommend", response_model=ConversationResponse)
async def recommend_products(query: UserQuery):
    """
    Endpoint principal de recomendación
    
    Flujo:
    1. Recibe consulta del usuario
    2. Procesa con LangGraph
    3. Retorna pregunta O recomendaciones
    """
    try:
        # Obtener o crear conversation_id
        conversation_id = query.conversation_id or str(uuid.uuid4())
        
        # Obtener historial
        history = conversations.get(conversation_id, [])
        
        # Procesar consulta
        result = conversation_service.process_query(
            query=query.query,
            conversation_history=history
        )
        
        # Actualizar historial
        history.append(f"Usuario: {query.query}")
        history.append(f"Sistema: {result['message']}")
        conversations[conversation_id] = history
        
        # Retornar respuesta
        return ConversationResponse(
            type=result['type'],
            message=result['message'],
            products=result['products'],
            conversation_id=conversation_id,
            question_count=result['question_count']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync-catalog")
async def sync_catalog(request: Request):
    """
    Sincroniza el catálogo desde Laravel
    
    Este endpoint lo llamará Laravel cuando haya cambios en productos
    """
    try:
        # Recibir productos del body
        body = await request.json()
        products = body.get('products', [])
        
        if not products:
            raise HTTPException(
                status_code=400,
                detail="No se recibieron productos en el body"
            )
        
        # Indexar en ChromaDB
        embedding_service.index_products(products)
        
        return {
            "message": f"Catálogo sincronizado: {len(products)} productos",
            "count": len(products)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error sincronizando catálogo: {str(e)}"
        )

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Elimina una conversación (reiniciar chat)"""
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"message": "Conversación eliminada"}

@app.get("/health")
async def health_check():
    """Health check para monitoreo"""
    return {
        "status": "healthy",
        "conversations_active": len(conversations),
        "llm_model": config.LLM_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Railway asigna el puerto dinámicamente
    port = int(os.getenv("PORT", 8001))
    
    uvicorn.run(app, host="0.0.0.0", port=port)