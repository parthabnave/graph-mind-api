from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

API_KEY = "AIzaSyCM6TmUkjA24zWM4ydkxFbRtDpoNEi4qNQ"
MODEL_NAME = "gemini-2.0-flash"

genai.configure(api_key=API_KEY)

class PlantUMLRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 1
    max_output_tokens: int = 2048

class PlantUMLResponse(BaseModel):
    plantuml_code: str | None
    error: str | None

def generate_plantuml(model_name, request: PlantUMLRequest):
    try:
        diagram_type = "chen" if "ER diagram" in request.prompt.lower() else "uml"
        plantuml_prompt = f"""
        Generate {diagram_type.upper()} code for the following description:
        {request.prompt}
        Please provide only the code block, enclosed in @start{diagram_type} and @end{diagram_type} tags.
        """

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            plantuml_prompt,
            generation_config=genai.GenerationConfig(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_output_tokens=request.max_output_tokens,
            ),
        )
        return response.text
    except Exception as e:
        logging.error(f"Error generating PlantUML: {e}")
        return None

def extract_plantuml_code(text, is_er_diagram):
    start_tag = "@startchen" if is_er_diagram else "@startuml"
    end_tag = "@endchen" if is_er_diagram else "@enduml"
    start_index = text.find(start_tag)
    end_index = text.find(end_tag)
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index + len(end_tag)]
    return None

@app.get("/")
async def health_check():
    return {"status": "API is running"}

@app.post("/generate_plantuml", response_model=PlantUMLResponse)
async def generate_plantuml_api(request: PlantUMLRequest = Body(...)):
    is_er_diagram = "ER diagram" in request.prompt.lower()
    generated_text = generate_plantuml(MODEL_NAME, request)
    if generated_text:
        extracted_code = extract_plantuml_code(generated_text, is_er_diagram)
        if extracted_code:
            return PlantUMLResponse(plantuml_code=extracted_code, error=None)
        else:
            return PlantUMLResponse(plantuml_code=None, error="Could not extract PlantUML or Chen code from the response.")
    raise HTTPException(status_code=500, detail="Failed to generate diagram code.")
