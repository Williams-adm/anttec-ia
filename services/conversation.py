from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import config
from services.embeddings import embedding_service
import json
import re

class ConversationState(TypedDict):
    """Estado de la conversación"""
    query: str
    conversation_history: List[str]
    extracted_requirements: dict
    question_count: int
    needs_clarification: bool
    products: List[dict]
    final_response: str

class ConversationService:  
    """Sistema de recomendación genérico - sin mapeos hardcodeados"""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            temperature=0.2
        )
        self.graph = self._create_graph()
    
    def _create_graph(self):
        workflow = StateGraph(ConversationState)
        
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("generate_question", self._generate_question)
        workflow.add_node("search_products", self._search_products)
        
        workflow.set_entry_point("analyze_query")
        
        workflow.add_conditional_edges(
            "analyze_query",
            self._should_ask_question,
            {
                "ask": "generate_question",
                "search": "search_products"
            }
        )
        
        workflow.add_edge("generate_question", END)
        workflow.add_edge("search_products", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: ConversationState) -> ConversationState:
        """Analiza la consulta - sin mapeos hardcodeados"""
        print("🧠 Analizando consulta...")
        
        system_prompt = """Eres un experto en periféricos de computadora.

Analiza la consulta del usuario y extrae información clave.

REGLAS PARA is_specific:
- is_specific = FALSE si SOLO dice: "quiero un mouse", "busco teclado", "necesito audífonos"
- is_specific = TRUE si menciona: uso (gaming/oficina/diseño), marca, modelo, o características técnicas

EXTRAE requisitos como texto libre:
- "requirements": lista de strings con CADA requisito mencionado
- Ejemplos: ["más de 20000 dpi", "inalámbrico", "para gaming", "marca Logitech", "precio menor a 100 dólares"]

Responde SOLO este JSON (sin markdown):
{
    "product_type": "tipo de producto o null",
    "requirements": ["requisito 1", "requisito 2"],
    "is_specific": true o false
}
"""
        
        # Construir mensaje con historial
        history_text = "\n".join(state['conversation_history'][-6:])
        
        if history_text:
            user_message = f"HISTORIAL DE CONVERSACIÓN:\n{history_text}\n\nNUEVA CONSULTA: {state['query']}"
        else:
            user_message = f"CONSULTA: {state['query']}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ])
            
            # Limpiar respuesta
            content = response.content.strip()
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = content.strip()
            
            print(f"📄 Respuesta LLM: {content[:200]}...")
            
            requirements = json.loads(content)
            
            # ARREGLAR: Asegurar que requirements sea siempre una lista
            if requirements.get('requirements') is None:
                requirements['requirements'] = []
            
            state['extracted_requirements'] = requirements
            
            # Determinar si necesita clarificación
            is_specific = requirements.get('is_specific', False)
            
            # Solo clarificar si es muy vago Y no hemos preguntado mucho
            state['needs_clarification'] = (
                not is_specific and 
                state['question_count'] < config.MAX_QUESTIONS
            )
            
            print(f"📋 Requisitos: {requirements}")
            print(f"❓ Necesita clarificación: {state['needs_clarification']}")
            
        except Exception as e:
            print(f"⚠️ Error en análisis: {e}")
            print(f"📄 Respuesta completa: {response.content if 'response' in locals() else 'N/A'}")
            
            # Fallback seguro
            state['needs_clarification'] = False
            state['extracted_requirements'] = {
                'product_type': self._extract_product_type_simple(state['query']),
                'requirements': [],
                'is_specific': False
            }
        
        return state
    
    def _extract_product_type_simple(self, query: str) -> str:
        """Extracción simple de tipo de producto (fallback)"""
        query_lower = query.lower()
        query = query.replace('á','a').replace('é','e').replace('í','i')

        keywords = {
            'mouse': [
                'mouse', 'ratón', 'raton', 'mause', 'maus',
                'mouse gamer', 'mouse gaming', 'mouse gamer pro',
                'mouse inalámbrico', 'mouse inalambrico', 'mouse wireless',
                'mouse bluetooth', 'mouse usb',
                'mouse óptico', 'mouse laser',
                'mouse rgb', 'mouse con luces',
                'mouse ergonómico', 'mouse ergonomico',
                'mouse vertical',
                'mouse para laptop', 'mouse para pc',
                'mouse profesional', 'mouse barato'
            ],
            'teclado': [
                'teclado', 'keyboard', 'keybord', 'teclao',
                'teclado gamer', 'teclado gaming',
                'teclado mecánico', 'teclado mecanico',
                'teclado de membrana',
                'teclado inalámbrico', 'teclado inalambrico', 'teclado wireless',
                'teclado bluetooth', 'teclado usb',
                'teclado rgb', 'teclado con luces',
                'teclado compacto', 'teclado 60%', 'teclado tkl',
                'teclado para laptop', 'teclado para pc',
                'teclado silencioso'
            ],   
            'audifonos': [
                'audífonos', 'audifonos', 'audifono',
                'auriculares', 'auricular',
                'headset', 'headset gamer', 'gaming headset',
                'cascos', 'cascos gamer',
                'audífonos gamer', 'audifonos gaming',
                'audífonos con micrófono', 'audifonos con microfono',
                'audífonos inalámbricos', 'audifonos inalambricos', 'wireless headset',
                'audífonos bluetooth',
                'audífonos usb', 'audífonos 3.5mm',
                'audífonos para pc', 'audífonos para laptop',
                'audífonos profesionales',
                'audífonos baratos'
            ],
            'monitor': [
                'monitor', 'pantalla', 'monitor pc', 'monitor computadora',
                'monitor gamer', 'monitor gaming',
                'monitor curvo', 'monitor plano',
                'monitor full hd', 'monitor 1080p',
                'monitor 2k', 'monitor 4k', 'monitor uhd',
                'monitor 144hz', 'monitor 165hz', 'monitor 240hz',
                'monitor ips', 'monitor va', 'monitor tn',
                'monitor para oficina', 'monitor profesional',
                'pantalla gaming'
            ],
            'webcam': [
                'webcam', 'web cam',
                'cámara web', 'camara web',
                'cámara para pc', 'camara para computadora',
                'webcam hd', 'webcam full hd', 'webcam 1080p',
                'webcam 4k',
                'webcam con micrófono', 'webcam con microfono',
                'webcam para streaming',
                'webcam para clases', 'webcam para reuniones',
                'cámara para zoom', 'camara para zoom'
            ],
            'parlantes': [
                'parlantes', 'altavoces', 'bocinas',
                'speakers', 'speakers pc',
                'parlantes gamer', 'parlantes gaming',
                'parlantes bluetooth',
                'parlantes usb',
                'parlantes para pc',
                'audio para computadora',
                'sistema de sonido'
            ],
            'extras': [
                'mousepad', 'pad de mouse', 'alfombrilla',
                'mousepad gamer', 'mouse pad grande',
                'control', 'gamepad', 'joystick',
                'volante gamer',
                'micrófono', 'microfono',
                'micrófono usb', 'microfono para streaming'
            ],                                                                       
        }
        
        for product_type, terms in keywords.items():
            if any(term in query_lower for term in terms):
                return product_type
        
        return None
    
    def _should_ask_question(self, state: ConversationState) -> str:
        """Decide si preguntar o buscar"""
        if state['question_count'] >= config.MAX_QUESTIONS:
            print(f"⚠️ Límite de preguntas ({config.MAX_QUESTIONS})")
            return "search"
        
        if state['needs_clarification']:
            print("❓ Generando pregunta...")
            return "ask"
        
        print("✅ Buscando productos...")
        return "search"
    
    def _generate_question(self, state: ConversationState) -> ConversationState:
        """Genera pregunta de clarificación"""
        print("❓ Generando pregunta...")
        
        system_prompt = """Eres un vendedor experto en periféricos de computadora.

El usuario ha hecho una consulta vaga. Genera UNA pregunta breve para clarificar.

PRIORIDADES:
1. Si no sabes el tipo de producto → "¿Qué tipo de periférico buscas?"
2. Si sabes el tipo pero no el uso → "¿Para gaming, trabajo o uso general?"
3. Si sabes uso pero no detalles → Pregunta presupuesto o marca

Responde SOLO la pregunta (1 línea).
"""
        
        context = f"Producto: {state['extracted_requirements'].get('product_type', 'desconocido')}\n"
        context += f"Requisitos actuales: {state['extracted_requirements'].get('requirements', [])}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ])
            
            state['final_response'] = response.content.strip()
            state['question_count'] += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            state['final_response'] = "¿Para qué uso lo necesitas principalmente?"
            state['question_count'] += 1
        
        return state
    
    def _search_products(self, state: ConversationState) -> ConversationState:
        """Busca productos usando IA para filtrar"""
        print("🔍 Buscando productos...")
        
        requirements = state['extracted_requirements']
        product_type_requested = requirements.get('product_type')
        
        # FASE 1: Búsqueda vectorial amplia
        search_query = self._build_search_query(requirements, state['query'])
        print(f"🔍 Query: '{search_query}'")
        
        try:
            # Traer MÁS productos para que el LLM filtre
            candidate_products = embedding_service.search(
                query=search_query,
                n_results=30  # Traer 30 candidatos
            )
            
            print(f"📊 Candidatos: {len(candidate_products)} productos")
            
            if not candidate_products:
                state['products'] = []
                state['final_response'] = "No encontré productos en el catálogo. ¿Podrías reformular tu búsqueda?"
                return state
            
            # ✨ NUEVA VALIDACIÓN: Verificar si el tipo de producto existe en el catálogo
            if product_type_requested:
                category_exists = self._product_type_exists_in_candidates(
                candidate_products, 
                product_type_requested
            )
            
            if not category_exists:
                print(f"⚠️ Categoría '{product_type_requested}' no existe en catálogo")
                state['products'] = []
                state['final_response'] = self._generate_category_not_found_message(product_type_requested)
                return state
            
            # FASE 2: LLM filtra y rankea los productos
            filtered_products = self._llm_filter_products(
                candidate_products,
                requirements,
                state['query']
            )
            
            print(f"✅ Después de filtro LLM: {len(filtered_products)} productos")
            
            # Limitar a PRODUCTS_TO_RETURN
            final_products = filtered_products[:config.PRODUCTS_TO_RETURN]
            
            state['products'] = final_products
            state['final_response'] = self._generate_recommendation(
                final_products,
                state['query'],
                requirements
            )
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            import traceback
            traceback.print_exc()
            
            state['products'] = []
            state['final_response'] = "Ocurrió un error al buscar productos. Por favor, intenta de nuevo."
        
        return state
    
    def _build_search_query(self, requirements: dict, original_query: str) -> str:
        """Construye query de búsqueda simple"""
        parts = []
        
        product_type = requirements.get('product_type')
        if product_type:
            parts.append(product_type)
        
        # Agregar requisitos como texto
        for req in requirements.get('requirements', []):
            if req:
                parts.append(req)
        
        return " ".join(parts) if parts else original_query
    
    def _llm_filter_products(
        self, 
        products: List[dict], 
        requirements: dict, 
        original_query: str
    ) -> List[dict]:
        """
        USA EL LLM PARA FILTRAR PRODUCTOS
        
        El LLM recibe los productos con TODAS sus especificaciones
        y decide cuáles cumplen los requisitos del usuario.
        
        NO HAY MAPEOS HARDCODEADOS - funciona con cualquier spec.
        """
        print("🤖 LLM filtrando productos...")
        
        system_prompt = """Eres un experto en periféricos de computadora.

Te daré una lista de productos con TODAS sus especificaciones técnicas.
Tu trabajo es FILTRAR y RANKEAR los productos según los requisitos del usuario.

REGLAS:
1. Lee TODAS las especificaciones de cada producto
2. Verifica si cumple con CADA requisito del usuario
3. Si dice "más de X", busca el valor en las specs y compara
4. Si dice "para gaming/oficina", analiza el uso previsto del producto
5. Si dice marca específica, verifica exactamente
6. Rankea por qué tan bien cumple los requisitos (no solo por similitud vectorial)

IMPORTANTE:
- Si el usuario dijo "para oficina", NO incluyas productos gaming
- Si el usuario dijo "para gaming", NO incluyas productos de oficina básicos
- Sé estricto con requisitos numéricos (DPI, peso, batería, etc.)

Responde SOLO un JSON (sin markdown):
{
    "filtered_products": [
        {
            "product_id": 1,
            "rank": 1,
            "match_score": 95,
            "reason": "Cumple todos los requisitos..."
        }
    ]
}

Ordena por match_score (mayor a menor).
Incluye SOLO productos que realmente cumplan (match_score >= 60).
"""
        
        # Preparar info de productos para el LLM
        products_data = []
        for p in products[:15]:  # Top 15 candidatos para no saturar el LLM
            product_info = {
                "id": p['id'],
                "name": p['name'],
                "brand": p['brand'],
                "category": p['category'],
                "subcategory": p['subcategory'],
                "description": p.get('description', ''),
                "specifications": p.get('specifications', []),
                "price_range": self._get_price_range(p),
                "similarity_score": p['similarity_score']
            }
            products_data.append(product_info)
        
        user_message = f"""CONSULTA DEL USUARIO: {original_query}

REQUISITOS EXTRAÍDOS:
- Tipo de producto: {requirements.get('product_type')}
- Requisitos específicos: {requirements.get('requirements', [])}

PRODUCTOS CANDIDATOS:
{json.dumps(products_data, indent=2, ensure_ascii=False)}

Filtra y rankea estos productos según los requisitos.
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ])
            
            # Limpiar y parsear
            content = response.content.strip()
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            result = json.loads(content)
            
            # Mapear IDs filtrados a productos originales
            filtered_ids = {item['product_id']: item for item in result.get('filtered_products', [])}
            
            filtered = []
            for product in products:
                if product['id'] in filtered_ids:
                    # Agregar info del LLM al producto
                    llm_info = filtered_ids[product['id']]
                    product['match_score'] = llm_info.get('match_score', 0)
                    product['match_reason'] = llm_info.get('reason', '')
                    filtered.append(product)
            
            # Ordenar por match_score del LLM
            filtered.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            
            print(f"🎯 LLM aprobó {len(filtered)} productos")
            
            return filtered
            
        except Exception as e:
            print(f"⚠️ Error en filtrado LLM: {e}")
            print(f"📄 Respuesta: {response.content if 'response' in locals() else 'N/A'}")
            
            # Fallback: retornar productos ordenados por similarity_score
            return sorted(products, key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    def _get_price_range(self, product: dict) -> str:
        """Obtiene rango de precios del producto"""
        variants = product.get('variants', [])
        if not variants:
            return "N/A"
        
        prices = [v['price'] for v in variants if 'price' in v]
        if not prices:
            return "N/A"
        
        min_price = min(prices)
        max_price = max(prices)
        
        if min_price == max_price:
            return f"${min_price:.2f}"
        else:
            return f"${min_price:.2f} - ${max_price:.2f}"
    
    def _generate_recommendation(
        self, 
        products: List[dict], 
        query: str, 
        requirements: dict
    ) -> str:
        """Genera respuesta final"""
        if not products:
            return "No encontré productos que cumplan con tus requisitos. ¿Podrías ajustar alguna especificación?"
        
        system_prompt = """Eres un vendedor experto en periféricos.

Presenta 2-3 productos de forma natural y conversacional.

Destaca POR QUÉ cada producto cumple con lo que buscan.
Sé breve (3-5 líneas).
NO uses listas con viñetas.
"""
        
        products_summary = f"Consulta: {query}\n\n"
        products_summary += "PRODUCTOS RECOMENDADOS:\n\n"
        
        for i, p in enumerate(products[:5], 1):
            products_summary += f"{i}. {p['name']} - {p['brand']}\n"
            products_summary += f"   Precio: {self._get_price_range(p)}\n"
            
            if p.get('match_reason'):
                products_summary += f"   Por qué: {p['match_reason']}\n"
            
            specs = p.get('specifications', [])[:3]
            if specs:
                specs_text = ", ".join([f"{s['name']}: {s['value']}" for s in specs])
                products_summary += f"   Specs: {specs_text}\n"
            
            products_summary += "\n"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=products_summary)
            ])
            
            return response.content.strip()
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            top = products[0]
            return f"Te recomiendo el {top['name']} de {top['brand']}. Es una excelente opción."
    
    def _product_type_exists_in_candidates(
        self, 
        candidates: List[dict], 
        product_type_requested: str
    ) -> bool:
        """
        Verifica si el tipo de producto solicitado existe en los candidatos
    
        Compara el product_type con:
        - category
        - subcategory
        - nombre del producto
    
        Retorna True si al menos 1 candidato parece ser del tipo correcto
        """
        product_type_lower = product_type_requested.lower()
    
        # Sinónimos comunes
        synonyms = {
            'mouse': ['mouse', 'ratón', 'raton'],
            'teclado': ['teclado', 'keyboard'],
            'audífonos': ['audífono', 'headset', 'auricular', 'audifonos'],
            'monitor': ['monitor', 'pantalla', 'display'],
            'router': ['ro  uter', 'enrutador'],
            'webcam': ['webcam', 'cámara web', 'camara'],
            'parlante': ['parlante', 'altavoz', 'bocina', 'speaker'],
        }
    
        # Obtener términos a buscar
        search_terms = [product_type_lower]
        for key, terms in synonyms.items():
            if product_type_lower in terms:
                search_terms = terms
                break
            
        # Buscar en los top 10 candidatos (más relevantes)
        for product in candidates[:10]:
            category = product.get('category', '').lower()
            subcategory = product.get('subcategory', '').lower()
            name = product.get('name', '').lower()

            # Verificar si algún término coincide
            for term in search_terms:
                if (term in category or 
                    term in subcategory or 
                    term in name):
                    return True

        return False

    def _generate_category_not_found_message(self, product_type: str) -> str:
        """ 
        Genera mensaje personalizado cuando una categoría no existe
        """
        messages = {    
            'monitor': "Lo siento, actualmente no tenemos monitores en nuestro catálogo.",
            'router': "Lo siento, actualmente no tenemos routers en nuestro catálogo.",
            'laptop': "Lo siento, actualmente no tenemos laptops en nuestro catálogo.",
            'parlante': "Lo siento, actualmente no tenemos parlantes en nuestro catálogo.",
        }

        # Mensaje específico o genérico
        product_type_lower = product_type.lower()

        for key, message in messages.items():
            if key in product_type_lower:
                return message

        # Mensaje genérico
        return f"Lo siento, no encontré {product_type}s en nuestro catálogo. Actualmente no lo contamos en nuestro catálogo. ¿Puedo ayudarte con algun otro producto?"

    def process_query(self, query: str, conversation_history: List[str] = None) -> dict:
        """Procesa una consulta"""
        try:
            initial_state = ConversationState(
                query=query,
                conversation_history=conversation_history or [],
                extracted_requirements={},
                question_count=len(conversation_history or []) // 2,
                needs_clarification=False,
                products=[],
                final_response=""
            )
            
            final_state = self.graph.invoke(initial_state)
            
            response_type = "question" if final_state['needs_clarification'] else "recommendation"
            
            return {
                "type": response_type,
                "message": final_state['final_response'],
                "products": final_state['products'] if response_type == "recommendation" else None,
                "question_count": final_state['question_count']
            }
            
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "type": "error",
                "message": "Error al procesar tu consulta. Intenta de nuevo.",
                "products": None,
                "question_count": 0
            }
        
# Instancia global
conversation_service = ConversationService()