import sys
sys.path.append(".secrets")

import logging
import os
from google.cloud import firestore
from google import genai
from google.cloud.exceptions import GoogleCloudError
from typing import Optional, Union
import hashlib
from google.genai.types import EmbedContentConfig
from google.cloud.firestore_v1.vector import Vector
from datetime import datetime
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.base_query import FieldFilter

from load_credentials import load_credentials

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FirestoreVectorStore:
    # Constante para el campo de embedding (fijo)
    EMBEDDING_KEY = "embedding"
    MAX_DIMENSION = 2048

    
    def __init__(self, project: str, database: str, collection: str, location: str = "us-east1"):
        """
        Inicializa el vector store de Firestore.
        
        Args:
            project: ID del proyecto de GCP
            database: Nombre de la base de datos de Firestore
            collection: Nombre de la colección donde se almacenarán los vectores
            location: Región de Google Cloud (default: "us-east1")
        """
        
        self.project = project
        self.collection = collection
        self.location = location
        
        # Configurar variables de entorno (con warnings si se sobrescriben)
        existing_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if existing_project and existing_project != project:
            logger.warning(f"GOOGLE_CLOUD_PROJECT ya establecido como '{existing_project}', sobrescribiendo con '{project}'")
        
        existing_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        if existing_location and existing_location != location:
            logger.warning(f"GOOGLE_CLOUD_LOCATION ya establecido como '{existing_location}', sobrescribiendo con '{location}'")
        
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        try:
            credentials = load_credentials()
            self.db = firestore.Client(project=project, database=database, credentials=credentials)
            self.genai_client = genai.Client(project=project, vertexai=True, credentials=credentials)
            logger.info(f"FirestoreVectorStore inicializado: proyecto={project}, database={database}, colección={collection}, location={location}")
        except GoogleCloudError as e:
            logger.error(f"Error al inicializar Firestore: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al inicializar: {e}")
            raise
    
    def _validate_dimension(self, dimension: Optional[int]) -> None:
        """
        Valida que la dimensión no exceda el máximo permitido.
        
        Args:
            dimension: Dimensión a validar
            
        Raises:
            ValueError: Si la dimensión excede el máximo permitido
        """
        if dimension is not None and dimension > self.MAX_DIMENSION:
            error_msg = f"La dimensión {dimension} excede el máximo permitido de {self.MAX_DIMENSION}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _generate_document_id(self, doc: dict, text_key: str) -> str:
        """
        Genera un ID único (hash) para un documento basado en su contenido.
        
        Args:
            doc: Documento del cual generar el ID
            text_key: Clave del campo de texto principal
            
        Returns:
            Hash SHA256 del contenido del documento (16 primeros caracteres)
        """
        # Usar el texto principal para generar el hash
        content = str(doc.get(text_key, ""))
        # Agregar otros campos relevantes para mayor unicidad
        for key, value in sorted(doc.items()):
            if key not in [self.EMBEDDING_KEY, 'id', 'created_at', 'updated_at']:
                content += f"{key}:{value}"
        
        # Generar hash SHA256 y tomar los primeros 16 caracteres
        hash_object = hashlib.sha256(content.encode('utf-8'))
        return hash_object.hexdigest()[:16]
    
    def embed_texts(self, texts: list[str], embedding_model: str, dimension: Optional[int] = None):
        """
        Genera embeddings para una lista de textos.
        Args:
            texts: Lista de textos a embedear
            embedding_model: Modelo de embedding a usar
            dimension: Dimensión del embedding (opcional, máximo 2048)
        Returns:
            Lista de objetos Vector con los embeddings
        Raises:
            ValueError: Si la dimensión excede el máximo permitido
        """
        try:
            # Validar dimensión
            self._validate_dimension(dimension)
            # Límite de batch para la API de embeddings (máximo 250)
            MAX_BATCH_SIZE = 250
            logger.info(f"Generando embeddings para {len(texts)} textos con modelo {embedding_model}")
            all_vectors = []
            total_batches = (len(texts) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            # Procesar en lotes de máximo 250
            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch_texts = texts[i:i + MAX_BATCH_SIZE]
                batch_num = (i // MAX_BATCH_SIZE) + 1
                logger.info(f"Procesando lote {batch_num}/{total_batches} ({len(batch_texts)} textos)")
                embeddings = self.genai_client.models.embed_content(
                    model=embedding_model,
                    contents=batch_texts,
                    config=EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=dimension
                    )
                )
                # Validar dimensión resultante
                if embeddings.embeddings:
                    actual_dimension = len(embeddings.embeddings[0].values)
                    if actual_dimension > self.MAX_DIMENSION:
                        error_msg = f"La dimensión resultante {actual_dimension} excede el máximo permitido de {self.MAX_DIMENSION}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                # Pasar a clase vector y agregar a la lista total
                batch_vectors = [Vector(value=embedding.values) for embedding in embeddings.embeddings]
                all_vectors.extend(batch_vectors)
                logger.info(f"Lote {batch_num}/{total_batches} completado: {len(batch_vectors)} vectores generados")
            logger.info(f"Embeddings generados exitosamente: {len(all_vectors)} vectores en total")
            return all_vectors
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error al generar embeddings: {e}")
            raise

    def embed_documents(self, documents: list[dict], embedding_model: str = "gemini-embedding-001", dimension: int = 2048, text_key: str = "text"):
        """
        Genera embeddings para una lista de documentos y los agrega al campo fijo 'embedding'.
        
        Args:
            documents: Lista de diccionarios con los documentos
            embedding_model: Modelo de embedding a usar (default: "gemini-embedding-001")
            dimension: Dimensión del embedding (default: 2048, máximo 2048)
            text_key: Clave del diccionario que contiene el texto a embedear (default: "text")
            
        Returns:
            Lista de documentos con el campo 'embedding' agregado
            
        Raises:
            ValueError: Si algún documento no tiene la clave de texto o si la dimensión excede el máximo
        """
        try:
            # Validar que todos los documentos tengan la clave de texto
            for i, doc in enumerate(documents):
                if text_key not in doc:
                    error_msg = f"El documento en índice {i} no tiene la clave '{text_key}' requerida"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Generar embeddings
            texts = [doc[text_key] for doc in documents]
            embeddings = self.embed_texts(
                texts=texts,
                embedding_model=embedding_model,
                dimension=dimension
            )
            
            # Agregar embeddings a los documentos (campo fijo "embedding")
            for i, doc in enumerate(documents):
                doc[self.EMBEDDING_KEY] = embeddings[i]
                logger.debug(f"Embedding agregado al documento {i}")
            
            logger.info(f"Embeddings agregados a {len(documents)} documentos")
            return documents
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error al embedear documentos: {e}")
            raise

    def add_documents(self, documents: list[dict], embedding_model: str = "gemini-embedding-001", dimension: int = 2048, text_key: str = "text"):
        """
        Agrega documentos con embeddings a la colección de Firestore.
        
        Args:
            documents: Lista de diccionarios con los documentos
            embedding_model: Modelo de embedding a usar (default: "gemini-embedding-001")
            dimension: Dimensión del embedding (default: 2048, máximo 2048)
            text_key: Clave del diccionario que contiene el texto a embedear (default: "text")
            
        Raises:
            ValueError: Si algún documento no tiene el campo 'embedding' o si hay errores de validación
            GoogleCloudError: Si hay errores al escribir en Firestore
        """
        try:
            # Generar embeddings si no existen
            documents_with_embeddings = self.embed_documents(
                documents=documents,
                embedding_model=embedding_model,
                dimension=dimension,
                text_key=text_key
            )
            
            # Verificar que todos tengan el campo embedding
            for i, doc in enumerate(documents_with_embeddings):
                if self.EMBEDDING_KEY not in doc:
                    error_msg = f"El documento en índice {i} no tiene el campo '{self.EMBEDDING_KEY}'"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Agregar documentos a la colección usando batch (máximo 500 por batch)
            BATCH_SIZE = 300
            total_docs = len(documents_with_embeddings)
            added_count = 0
            
            # Dividir en lotes de 300 documentos
            for i in range(0, total_docs, BATCH_SIZE):
                batch = self.db.batch()
                batch_docs = documents_with_embeddings[i:i + BATCH_SIZE]
                
                for doc in batch_docs:
                    # Agregar metadatos obligatorios
                    doc_id = self._generate_document_id(doc, text_key)
                    doc['id'] = doc_id
                    doc['created_at'] = datetime.utcnow().isoformat()
                    
                    # Usar el hash como document ID en Firestore
                    doc_ref = self.db.collection(self.collection).document(doc_id)
                    batch.set(doc_ref, doc)
                
                batch.commit()
                added_count += len(batch_docs)
                logger.info(f"Lote agregado: {len(batch_docs)} documentos. Total: {added_count}/{total_docs}")
            
            logger.info(f"{total_docs} documentos agregados exitosamente a la colección '{self.collection}'")
            
        except ValueError:
            raise
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al agregar documentos: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al agregar documentos: {e}")
            raise

    def count_documents(self):
        """
        Cuenta los documentos en la colección.

        Returns:
            Número de documentos en la colección

        Raises:
            GoogleCloudError: Si hay errores al consultar Firestore
            ValueError: Si el resultado tiene formato inesperado
        """
        try:
            count_query = self.db.collection(self.collection).count()
            count_result = count_query.get()
            
            # QueryResultsList es iterable - extraer el primer resultado de la agregación
            for aggregation_result in count_result:
                # Cada resultado de agregación es una lista con el valor del count
                if isinstance(aggregation_result, list) and len(aggregation_result) > 0:
                    count = aggregation_result[0].value
                    logger.info(f"Cantidad de documentos en '{self.collection}': {count}")
                    return count
            
            # Si llegamos aquí, formato inesperado
            error_msg = f"Formato inesperado del resultado de count: {type(count_result)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al contar documentos: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al contar documentos: {e}")
            raise
    
    def get_documents(self):
        """
        Obtiene todos los documentos de la colección.
        
        Returns:
            Lista de diccionarios con los documentos
            
        Raises:
            GoogleCloudError: Si hay errores al consultar Firestore
        """
        try:
            docs = [doc.to_dict() for doc in self.db.collection(self.collection).stream()]
            logger.info(f"Obtenidos {len(docs)} documentos de la colección '{self.collection}'")
            return docs
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al obtener documentos: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al obtener documentos: {e}")
            raise

    def get_document_by_key(self, key: str, value: Union[str, list, dict]):
        """
        Busca documentos por una llave y valor específicos.
        
        Args:
            key: Nombre del campo a buscar
            value: Valor a buscar
            
        Returns:
            Lista de diccionarios con los documentos encontrados
            
        Raises:
            GoogleCloudError: Si hay errores al consultar Firestore
        """
        try:
            docs = self.db.collection(self.collection).where(key, "==", value).get()
            result = [doc.to_dict() for doc in docs]
            logger.info(f"Encontrados {len(result)} documentos con {key}={value}")
            return result
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al buscar documentos por llave: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al buscar documentos: {e}")
            raise

    def get_document_by_id(self, document_id: str):
        """
        Obtiene un documento específico por su ID.
        
        Args:
            document_id: ID único del documento (hash)
            
        Returns:
            Diccionario con el documento encontrado o None si no existe
            
        Raises:
            GoogleCloudError: Si hay errores al consultar Firestore
        """
        try:
            doc_ref = self.db.collection(self.collection).document(document_id)
            doc_snapshot = doc_ref.get()
            
            if doc_snapshot.exists:
                result = doc_snapshot.to_dict()
                logger.info(f"Documento encontrado con ID: {document_id}")
                return result
            else:
                logger.warning(f"No se encontró documento con ID: {document_id}")
                return None
                
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al buscar documento por ID: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al buscar documento: {e}")
            raise
    
    def delete_document_by_key(self, key: str, value: Union[str, list, dict], batch_size: int = 500):
        """
        Elimina documentos por llave y valor específicos.
        
        ADVERTENCIA: Esta operación es irreversible.
        
        Args:
            key: Nombre del campo a buscar
            value: Valor a buscar
            batch_size: Número de documentos a eliminar por lote (máximo 500)
            
        Returns:
            Número de documentos eliminados
            
        Raises:
            ValueError: Si batch_size excede 500 o no se encuentran documentos
            GoogleCloudError: Si hay errores al eliminar en Firestore
        """
        try:
            if batch_size > 500:
                error_msg = "batch_size no puede exceder 500 (límite de Firestore)"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Buscar documentos que coincidan
            doc_refs = self.db.collection(self.collection).where(key, "==", value).stream()
            doc_refs_list = list(doc_refs)
            
            if not doc_refs_list:
                logger.warning(f"No se encontraron documentos con {key}={value}")
                return 0
            
            logger.warning(f"Eliminando {len(doc_refs_list)} documentos con {key}={value}")
            
            # Eliminar en batches
            deleted_count = 0
            total_docs = len(doc_refs_list)
            
            for i in range(0, total_docs, batch_size):
                batch = self.db.batch()
                batch_refs = doc_refs_list[i:i + batch_size]
                
                for doc_ref in batch_refs:
                    batch.delete(doc_ref.reference)
                
                batch.commit()
                deleted_count += len(batch_refs)
                logger.info(f"Eliminados {len(batch_refs)} documentos. Total: {deleted_count}/{total_docs}")
            
            logger.info(f"Eliminación completada: {deleted_count} documentos con {key}={value}")
            return deleted_count
            
        except ValueError:
            raise
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al eliminar documentos por llave: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al eliminar documentos: {e}")
            raise
    
    def delete_document_by_id(self, document_id: str):
        """
        Elimina un documento específico por su ID.
        
        ADVERTENCIA: Esta operación es irreversible.
        
        Args:
            document_id: ID único del documento (hash)
            
        Returns:
            True si se eliminó, False si no se encontró
            
        Raises:
            GoogleCloudError: Si hay errores al eliminar en Firestore
        """
        try:
            doc_ref = self.db.collection(self.collection).document(document_id)
            doc_snapshot = doc_ref.get()
            
            if not doc_snapshot.exists:
                logger.warning(f"No se encontró documento con ID: {document_id}")
                return False
            
            logger.warning(f"Eliminando documento con ID: {document_id}")
            doc_ref.delete()
            logger.info(f"Documento {document_id} eliminado exitosamente")
            return True
            
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al eliminar documento por ID: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al eliminar documento: {e}")
            raise
    
    def update_document_by_key(self, key: str, value: Union[str, list, dict], 
                               embedding_model: str = "gemini-embedding-001", dimension: int = 2048, 
                               text_key: str = "text", update_fields: Optional[dict] = None):
        """
        Actualiza documentos por llave, regenerando embeddings y opcionalmente otros campos.
        
        Args:
            key: Nombre del campo a buscar
            value: Valor a buscar
            embedding_model: Modelo de embedding a usar (default: "gemini-embedding-001")
            dimension: Dimensión del embedding (default: 2048, máximo 2048)
            text_key: Clave del diccionario que contiene el texto a embedear (default: "text")
            update_fields: Diccionario opcional con campos adicionales a actualizar
            
        Returns:
            Número de documentos actualizados
            
        Raises:
            ValueError: Si no se encuentran documentos o si hay errores de validación
            GoogleCloudError: Si hay errores al actualizar en Firestore
        """
        try:
            # Buscar documentos por llave
            doc_refs = self.db.collection(self.collection).where(key, "==", value).get()
            
            if not doc_refs:
                error_msg = f"No se encontraron documentos con la llave '{key}' y valor '{value}'"
                logger.warning(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Encontrados {len(doc_refs)} documentos para actualizar")
            
            # Convertir a diccionarios para procesar
            docs = [doc_ref.to_dict() for doc_ref in doc_refs]
            
            # Validar que todos tengan la clave de texto
            for i, doc in enumerate(docs):
                if text_key not in doc:
                    error_msg = f"El documento en índice {i} no tiene la clave '{text_key}' requerida"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Generar nuevos embeddings
            texts = [doc[text_key] for doc in docs]
            embeddings = self.embed_texts(
                texts=texts,
                embedding_model=embedding_model,
                dimension=dimension
            )
            
            # Actualizar documentos en Firestore
            batch = self.db.batch()
            updated_count = 0
            
            for i, doc_ref in enumerate(doc_refs):
                # Actualizar el campo embedding (fijo)
                update_data = {
                    self.EMBEDDING_KEY: embeddings[i].values,
                    'updated_at': datetime.utcnow().isoformat()
                }
                
                # Agregar campos adicionales si se proporcionan
                if update_fields:
                    update_data.update(update_fields)
                
                batch.update(doc_ref, update_data)
                updated_count += 1
            
            batch.commit()
            logger.info(f"{updated_count} documentos actualizados exitosamente")
            return updated_count
            
        except ValueError:
            raise
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al actualizar documentos: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al actualizar documentos: {e}")
            raise

    def update_document_by_id(self, document_id: str, updated_data: dict, 
                              embedding_model: str = "gemini-embedding-001", dimension: int = 2048, 
                              text_key: str = "text"):
        """
        Actualiza un documento específico por su ID, regenerando el embedding.
        
        Args:
            document_id: ID único del documento (hash)
            updated_data: Diccionario con los campos a actualizar
            embedding_model: Modelo de embedding a usar (default: "gemini-embedding-001")
            dimension: Dimensión del embedding (default: 2048, máximo 2048)
            text_key: Clave del diccionario que contiene el texto a embedear (default: "text")
            
        Raises:
            ValueError: Si el documento no existe o no tiene el campo de texto
            GoogleCloudError: Si hay errores al actualizar en Firestore
        """
        try:
            # Obtener referencia del documento por ID
            doc_ref = self.db.collection(self.collection).document(document_id)
            doc_snapshot = doc_ref.get()
            
            if not doc_snapshot.exists:
                error_msg = f"No se encontró documento con ID '{document_id}'"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Actualizando documento con ID: {document_id}")
            
            # Obtener datos actuales y mezclar con los nuevos
            current_data = doc_snapshot.to_dict()
            current_data.update(updated_data)
            
            # Validar que tenga el campo de texto
            if text_key not in current_data:
                error_msg = f"El documento no tiene la clave '{text_key}' requerida"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Generar nuevo embedding
            embeddings = self.embed_texts(
                texts=[current_data[text_key]],
                embedding_model=embedding_model,
                dimension=dimension
            )
            
            # Preparar datos de actualización
            update_data = {
                self.EMBEDDING_KEY: embeddings[0],
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Agregar los campos actualizados
            update_data.update(updated_data)
            
            # Actualizar el documento
            doc_ref.update(update_data)
            logger.info(f"Documento {document_id} actualizado exitosamente")
            
        except ValueError:
            raise
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al actualizar documento: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al actualizar documento: {e}")
            raise
    
    def delete_collection(self, batch_size: int = 500):
        """
        Elimina todos los documentos de la colección.
        
        ADVERTENCIA: Esta operación es irreversible y eliminará todos los documentos.
        
        Args:
            batch_size: Número de documentos a eliminar por lote (máximo 500)
            
        Returns:
            Número total de documentos eliminados
            
        Raises:
            ValueError: Si batch_size excede 500
            GoogleCloudError: Si hay errores al eliminar en Firestore
        """
        try:
            if batch_size > 500:
                error_msg = "batch_size no puede exceder 500 (límite de Firestore)"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.warning(f"Iniciando eliminación de todos los documentos en la colección '{self.collection}'")
            
            deleted_count = 0
            collection_ref = self.db.collection(self.collection)
            
            while True:
                # Obtener un lote de documentos
                docs = list(collection_ref.limit(batch_size).stream())
                
                if not docs:
                    break  # No hay más documentos
                
                # Eliminar el lote usando batch
                batch = self.db.batch()
                for doc in docs:
                    batch.delete(doc.reference)
                
                batch.commit()
                deleted_count += len(docs)
                logger.info(f"Eliminados {len(docs)} documentos. Total: {deleted_count}")
            
            logger.info(f"Colección '{self.collection}' limpiada exitosamente. Total eliminados: {deleted_count}")
            return deleted_count
            
        except ValueError:
            raise
        except GoogleCloudError as e:
            logger.error(f"Error de Firestore al eliminar colección: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al eliminar colección: {e}")
            raise

    def as_retriever(self, query: str, k: int = 5, model: str = "gemini-embedding-001", 
                    dimension: int = 2048, distance_measure: DistanceMeasure = DistanceMeasure.COSINE, 
                    filters: dict = None):
        """
        Búsqueda de similaridad vectorial optimizada con filtros opcionales.
        
        Args:
            query: Texto de búsqueda
            k: Número de resultados a retornar
            model: Modelo de embeddings a usar
            dimension: Dimensión de los embeddings
            distance_measure: Medida de distancia (COSINE, EUCLIDEAN, DOT_PRODUCT)
            filters: Diccionario de filtros para aplicar condiciones.
                    Soporta dos formatos:
                    - Igualdad simple: {"campo": valor}
                    - Operador IN: {"campo": ("in", [valor1, valor2, ...])}
                    
                    Ejemplos:
                    - filters={"categoria": "cardiologia", "activo": True}
                    - filters={"categoria": ("in", ["cardiologia", "neurologia"]), "activo": True}
                    
                    NOTA: Si una lista 'in' tiene más de 30 valores, se divide automáticamente
                    en múltiples consultas y se combinan los resultados.
        
        Returns:
            Lista de documentos similares con sus scores
            
        Raises:
            ValueError: Si hay errores en el formato de filtros
        """
        
        try:
            # Generar embedding del query
            vector_list = self.embed_texts(texts=[query], embedding_model=model, dimension=dimension)
            query_vector = vector_list[0]
            
            # Identificar si hay filtros IN con más de 30 valores
            has_large_in_filter = False
            large_in_field = None
            large_in_values = None
            other_filters = {}
            has_other_filters = False
            
            if filters:
                for field, value in filters.items():
                    if isinstance(value, tuple) and len(value) == 2 and value[0] == "in":
                        operator, values_list = value
                        
                        # Validaciones básicas
                        if not isinstance(values_list, list):
                            error_msg = f"El operador 'in' requiere una lista de valores, recibido: {type(values_list)}"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                        
                        if len(values_list) == 0:
                            error_msg = "El operador 'in' requiere al menos un valor en la lista"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                        
                        # Si tiene más de 30 valores, lo manejamos especialmente
                        if len(values_list) > 30:
                            if has_large_in_filter:
                                error_msg = "Solo se puede tener un filtro 'in' con más de 30 valores por consulta"
                                logger.error(error_msg)
                                raise ValueError(error_msg)
                            
                            has_large_in_filter = True
                            large_in_field = field
                            large_in_values = values_list
                            logger.info(f"Filtro IN grande detectado: {field} con {len(values_list)} valores. Se dividirá en chunks.")
                        else:
                            other_filters[field] = value
                            has_other_filters = True
                    else:
                        other_filters[field] = value
                        has_other_filters = True
            
            # Determinar el chunk_size óptimo basado en si hay otros filtros
            # Si hay otros filtros + filtro IN grande, reducimos el chunk_size para evitar el límite de 30 disjunciones
            if has_large_in_filter and has_other_filters:
                # Con otros filtros, usamos chunks más pequeños para evitar el límite de disjunciones
                chunk_size = 10
                logger.info(f"Usando chunk_size={chunk_size} debido a filtros adicionales (evitar límite de 30 disjunciones)")
            else:
                chunk_size = 30
            
            # Función auxiliar para ejecutar una búsqueda con filtros específicos
            def _execute_search(search_filters, limit_override=None):
                collection_ref = self.db.collection(self.collection)
                
                # Aplicar filtros
                for field, value in search_filters.items():
                    if isinstance(value, tuple) and len(value) == 2 and value[0] == "in":
                        operator, values_list = value
                        collection_ref = collection_ref.where(filter=FieldFilter(field, "in", values_list))
                        logger.debug(f"Filtro IN aplicado: {field} in {values_list[:3]}... ({len(values_list)} valores)")
                    else:
                        collection_ref = collection_ref.where(filter=FieldFilter(field, "==", value))
                        logger.debug(f"Filtro de igualdad aplicado: {field} == {value}")
                
                # Búsqueda vectorial con límite ajustado
                search_limit = limit_override if limit_override else k
                response = collection_ref.find_nearest(
                    vector_field=self.EMBEDDING_KEY,
                    query_vector=query_vector,
                    distance_measure=distance_measure,
                    limit=search_limit
                )
                
                return response.get()
            
            # Si NO hay filtro IN grande, ejecutar búsqueda normal
            if not has_large_in_filter:
                results = _execute_search(other_filters if filters else {})
                
                result_list = [
                    {
                        **{k: v for k, v in doc.to_dict().items() 
                        if k not in [self.EMBEDDING_KEY, "vector_distance"]},
                        "score": 1 - doc.to_dict().get("vector_distance", 0)
                    }
                    for doc in results
                ]
                
                logger.info(f"Búsqueda completada: {len(result_list)} resultados encontrados")
                return result_list
            
            # Si HAY filtro IN grande, dividir en chunks y combinar resultados
            logger.info(f"Ejecutando búsqueda en múltiples chunks para {large_in_field}")
            
            all_results = []
            num_chunks = (len(large_in_values) - 1) // chunk_size + 1
            
            # Solicitar más resultados por chunk para compensar
            # Pedimos k * (número de chunks) para asegurar que tenemos suficientes
            results_per_chunk = max(k * 2, k * num_chunks // 2)
            
            # Dividir la lista en chunks
            for i in range(0, len(large_in_values), chunk_size):
                chunk = large_in_values[i:i + chunk_size]
                
                # Crear filtros para este chunk
                chunk_filters = other_filters.copy()
                chunk_filters[large_in_field] = ("in", chunk)
                
                # Ejecutar búsqueda para este chunk con límite aumentado
                logger.info(f"Ejecutando chunk {i//chunk_size + 1}/{num_chunks} con {len(chunk)} valores (solicitando {results_per_chunk} resultados)")
                
                try:
                    chunk_results = _execute_search(chunk_filters, limit_override=results_per_chunk)
                    
                    # Agregar resultados con sus scores
                    for doc in chunk_results:
                        doc_dict = doc.to_dict()
                        all_results.append({
                            **{k: v for k, v in doc_dict.items() 
                            if k not in [self.EMBEDDING_KEY, "vector_distance"]},
                            "score": 1 - doc_dict.get("vector_distance", 0),
                            "_doc_id": doc.id  # Para deduplicar
                        })
                except Exception as chunk_error:
                    logger.warning(f"Error en chunk {i//chunk_size + 1}: {chunk_error}. Continuando con otros chunks...")
                    continue
            
            if not all_results:
                logger.warning("No se obtuvieron resultados de ningún chunk")
                return []
            
            # Deduplicar resultados (por si un documento aparece en múltiples chunks)
            unique_results = {}
            for result in all_results:
                doc_id = result.pop("_doc_id")
                if doc_id not in unique_results or result["score"] > unique_results[doc_id]["score"]:
                    unique_results[doc_id] = result
            
            # Ordenar por score descendente y tomar los top k
            final_results = sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:k]
            
            logger.info(f"Búsqueda con chunks completada: {len(final_results)} resultados finales de {len(all_results)} totales (después de deduplicación)")
            return final_results
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            raise