import sys
sys.path.append(".secrets")

import pandas as pd
from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from load_credentials import load_credentials


# --- Pydantic models ---

class InsuranceProduct(BaseModel):
    id: int = Field(description="Unique identifier for the product")
    name: str = Field(description="Name of the insurance product")
    short_description: str = Field(description="A brief one-line description")
    complete_description: str = Field(description="A detailed multi-sentence description")


class InsuranceProductList(BaseModel):
    products: list[InsuranceProduct] = Field(description="List of 10 insurance products")


# --- Parser and prompt ---

parser = JsonOutputParser(pydantic_object=InsuranceProductList)

prompt = PromptTemplate(
    template=(
        "You are a creative insurance product designer. "
        "Generate 10 completely imaginary and unreal insurance products. "
        "They should be fun, creative, and obviously fictional. "
        "All of the names and descriptions must be in spanish. "
        "Each product must have: id, name, short_description, and complete_description.\n\n"
        "{format_instructions}\n"
    ),
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# --- LLM call ---

credentials = load_credentials()

llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    credentials=credentials,
)

chain = prompt | llm | parser

result = chain.invoke({})

# --- Build DataFrame ---

df = pd.DataFrame(result["products"])
df.to_pickle("./output_files/insurance_products.pkl")
print(df.to_string(index=False))
