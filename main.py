import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from typing import Optional
from pydantic import BaseModel, Field
import pkg_resources

from utils.main_function import call_agente_workshop



app = FastAPI()


# Configuración de CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputAgentCall(BaseModel):
    thread_id: str = Field(example="test_thread_id", description="ID del hilo de la solicitud.")
    texto_usuario: str = Field(example="Quiero saber si me cubre ...", description="El texto input de la conversación por parte del usuario")


descripcion_path = 'Endpoint que lleva la conversación del agente'
endpoint_path = '/call-agent'
summary_path = 'Endpoint de llamada al agente'
@app.post(endpoint_path, summary=summary_path, description=descripcion_path)
async def call_agent(input: InputAgentCall):

    response = call_agente_workshop(thread_id=input.thread_id, user_message=input.texto_usuario)

    return response


@app.get("/version")
def get_versions():
    """Return versions of all installed packages"""
    installed_packages = pkg_resources.working_set
    packages_dict = {pkg.key: pkg.version for pkg in installed_packages}
    return packages_dict


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"


@app.options("/options-consulta-punto-com")
async def options_product_recomm(request: Request):
    print('#################################  LLEGÓ A OPTIONS  #################################')
    return {}



if __name__ == '__main__':
    import uvicorn
    puerto = os.environ.get("PORT", 8080)
    # puerto = 8001
    uvicorn.run(app, host="127.0.0.1", port=int(puerto))