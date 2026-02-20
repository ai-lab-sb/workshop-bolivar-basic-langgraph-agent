from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple
from google.cloud import firestore
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple, ChannelVersions
import pickle
from datetime import datetime
import pytz


class JsonPlusSerializerCompat(JsonPlusSerializer):
    #Clase para serializar-deserializar el checkpointer, hereda métodos de JsonPlusSerializer
    def loads(self, data: bytes) -> Any:
        if data.startswith(b"\x80") and data.endswith(b"."):
            return pickle.loads(data)
        return super().loads(data)
    
class FirestoreSaver(BaseCheckpointSaver):
    """
    Clase para implementar memoria de Langgraph en Firestore, debe especificarse 
    la base de datos (database) el nombre de coleccion de los checkpoints (collection_name)
    y el nombre de la coleccion de pasos intermedios (pw_collection_name).
    """
    serde = JsonPlusSerializerCompat()

    def __init__(self, database = "(default)", collection_name: str = "checkpoints", pw_collection_name: str = "checkpoint_writes", serde: Optional[Any] = None) -> None:
        super().__init__(serde=serde)
        self.db: firestore.Client = firestore.Client(database = database)
        self.async_db: firestore.AsyncClient = firestore.AsyncClient(database = database)
        self.collection_name: str = collection_name
        self.pw_collection_name: str = pw_collection_name

    # Método para traer memoria asociada a un thread_id
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id: str = config["configurable"]["thread_id"]
        thread_ts: Optional[str] = config["configurable"].get("thread_ts")
        
        doc_ref: firestore.DocumentReference = self.db.collection(self.collection_name).document(thread_id)
        doc: firestore.DocumentSnapshot = doc_ref.get()
        #Trae todo lo asociado al thead_id

        if not doc.exists:
            return None

        data: Dict[str, Any] = doc.to_dict()
        return self._process_checkpoint_data_common(data)

    # Método asincronico para traer memmoria
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id: str = config["configurable"]["thread_id"]
        thread_ts: Optional[str] = config["configurable"].get("thread_ts")
        
        doc_ref: firestore.AsyncDocumentReference = self.async_db.collection(self.collection_name).document(thread_id)
        doc: firestore.DocumentSnapshot = await doc_ref.get()
        
        data: Dict[str, Any] = doc.to_dict()
        return await self._process_checkpoint_data_common(data)

    # Para listar checkpoints (Para listar checkpoints basados en un criterio)
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id: Optional[str] = config["configurable"]["thread_id"] if config else None
        if filter:
            raise NotImplementedError("No se cuenta con la funcionalidad de filtrado")
        
        # Obtiene una referencia a la colección de checkpoints
        col_ref: firestore.CollectionReference = self.db.collection(self.collection_name)
        
        # Si se proporcionó un thread_id, filtra por ese thread_id
        if thread_id:
            col_ref = col_ref.where("thread_id", "==", thread_id)
        
        docs: firestore.QuerySnapshot = col_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit or 100).get()
        
        for doc in docs:
            yield self._process_checkpoint_data_common(doc.to_dict())

    # Método asincronico  para listar checkpoints (Para listar checkpoints basados en un criterio)
    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id: Optional[str] = config["configurable"]["thread_id"] if config else None
        if filter:
            raise NotImplementedError("Filtering is not implemented for FirestoreSaver")
        
        # Obtiene una referencia a la colección de checkpoints
        col_ref: firestore.AsyncCollectionReference = self.async_db.collection(self.collection_name)
        
        # Si se proporcionó un thread_id, filtra por ese thread_id
        if thread_id:
            col_ref = col_ref.where("thread_id", "==", thread_id)
        
        docs: firestore.QuerySnapshot = await col_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit or 100).get()
        
        async for doc in docs:
            yield self._process_checkpoint_data_common(doc.to_dict())

    # Para guardar un checkpoint
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id: str = config["configurable"]["thread_id"]
        timestamp: str = datetime.now(pytz.timezone('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S')
        ts: str = checkpoint["id"]
        
        doc_ref: firestore.DocumentReference = self.db.collection(self.collection_name).document(thread_id)
        doc_ref.set({
            "checkpoint": self.serde.dumps(checkpoint),
            "metadata": self.serde.dumps(metadata),
            "thread_id": thread_id,
            "timestamp": timestamp
        })

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": ts,
            },
        }

    # Método asincronico para put
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id: str = config["configurable"]["thread_id"]
        timestamp: str = datetime.now(pytz.timezone('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S')
        ts: str = checkpoint["id"]
        
        doc_ref: firestore.AsyncDocumentReference = self.async_db.collection(self.collection_name).document(f"{thread_id}_{timestamp}")
        await doc_ref.set({
            "checkpoint": self.serde.dumps(checkpoint),
            "metadata": self.serde.dumps(metadata),
            "thread_id": thread_id,
            "timestamp": timestamp
        })

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": ts,
            },
        }
    
    def put_writes(
        self,
        config: dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Guarda escrituras intermedias vinculados a un checkpoint.


        Args:
            config (dict): Configuración del checkpoint.
            writes (Sequence[Tuple[str, Any]]): Lista de escrituras intermedias, cada uno como una pareja (channel, value).
            task_id (str): Identificador de la tarea creando las escrituras intermedias.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        
        for idx, (channel, value) in enumerate(writes):
            doc_id = f"{thread_id}"  # Documento para esta base
            type_, serialized_value = self.serde.dumps_typed(value)

            write_data = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "channel": channel,
                "type": type_,
                "value": serialized_value,
            }

            # Guardado de documento
            self.db.collection(self.pw_collection_name).document(doc_id).set(write_data, merge=True)

    def _process_checkpoint_data_common(self, data: Dict[str, Any]) -> CheckpointTuple:
        checkpoint: Checkpoint = self.serde.loads(data["checkpoint"])
        metadata: CheckpointMetadata = self.serde.loads(data["metadata"])
        thread_id: str = data["thread_id"]
        thread_ts: str = data["timestamp"]

        config: RunnableConfig = {"configurable": {"thread_id": thread_id, "thread_ts": thread_ts}}
        return CheckpointTuple(config=config, checkpoint=checkpoint, metadata=metadata, parent_config=None)