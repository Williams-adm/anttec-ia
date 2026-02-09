from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ProductSpecification(BaseModel):
    """Especificación de producto (ej: DPI: 16000)"""
    name: str
    value: str

class ProductVariant(BaseModel):
    """Variante de producto (ej: Color rojo, $29.99)"""
    id: int
    sku: str
    price: float
    stock: int
    features: List[Dict[str, Any]]  # Color, tamaño, etc.

class Product(BaseModel):
    """Producto completo"""
    id: int
    name: str
    model: str
    description: Optional[str]
    brand: str
    category: str
    subcategory: str
    specifications: List[ProductSpecification]
    variants: List[ProductVariant]

class UserQuery(BaseModel):
    """Consulta del usuario"""
    query: str
    conversation_id: Optional[str] = None

class ConversationResponse(BaseModel):
    """Respuesta del sistema"""
    type: str  # "question" o "recommendation"
    message: str
    products: Optional[List[Dict[str, Any]]] = None
    conversation_id: str
    question_count: int