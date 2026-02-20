# workshop-bolivar-basic-langgraph-agent
Este repositorio contiene el material y la estructura de código del workshop de Inteligencia Artificial, diseñado para construir un agente de IA con capacidades de búsqueda semántica sobre una base de conocimiento propia.

## Contenido del repositorio

El repositorio está organizado de la siguiente manera:

- **`helpers/`** — Módulos auxiliares con la configuración de clientes y modelos de GCP, y los prompts del agente.
- **`utils/`** — Lógica principal del agente, incluyendo la configuración del Vector Store en Firestore, la memoria, las tools y la función principal de ejecución.
- **`output_files/`** — Archivos generados durante la ejecución del agente.
- **`main.py`** — Punto de entrada de la aplicación.
- **`Dockerfile`** — Imagen de contenedor para el despliegue.
- **`cloudbuild.yaml`** — Pipeline de CI/CD para despliegue automático en GCP.
- **`requirements.txt`** — Dependencias del proyecto.

## Despliegue

La aplicación está diseñada para ser desplegada en **Google Cloud Platform** usando **Cloud Build** como pipeline de CI/CD y **Cloud Run** como plataforma de ejecución. Cualquier cambio en la rama principal dispara automáticamente el pipeline definido en `cloudbuild.yaml`, que construye la imagen Docker y la despliega en Cloud Run.

## Material del workshop

El notebook interactivo del workshop está disponible en Google Colab. Contiene toda la teoría y el código necesario para entender y construir el agente paso a paso, sin necesidad de configurar ningún entorno local.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ijfA__7N7ZLESS2rGia5y9-QYltRKMxM?usp=sharing)