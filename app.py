from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

API_KEY = "AIzaSyCM6TmUkjA24zWM4ydkxFbRtDpoNEi4qNQ"
MODEL_NAME = "gemini-2.0-flash"

genai.configure(api_key=API_KEY)

class PlantUMLRequest(BaseModel):
    prompt: str

class PlantUMLResponse(BaseModel):
    plantuml_code: str | None
    error: str | None

def generate_plantuml(model_name: str, prompt: str) -> str | None:
    try:
        # Use AI to identify the diagram type and generate PlantUML
        classification_prompt = f"Identify the type of diagram (Use Case, Architecture, or Other) based on this description: {prompt}"
        model = genai.GenerativeModel(model_name)
        diagram_type_response = model.generate_content(classification_prompt)
        diagram_type = diagram_type_response.text.strip().lower()

        # Formulating the generation prompt
        plantuml_prompt = f"""
        Generate PlantUML code for a {diagram_type.capitalize()} Diagram based on this description:
        {prompt}
        - Enclose the code in @startuml and @enduml tags only.
        - Assume required components to generate accurate code
        - Do not include explanations or additional text outside the tags.
        """

        response = model.generate_content(
            plantuml_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        return response.text
    except Exception as e:
        print(f"Error generating PlantUML: {e}")
        return None

def extract_plantuml_code(text: str) -> str | None:
    start_tag = "@startuml"
    end_tag = "@enduml"
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
    generated_text = generate_plantuml(MODEL_NAME, request.prompt)
    if generated_text:
        extracted_code = extract_plantuml_code(generated_text)
        if extracted_code:
            return PlantUMLResponse(plantuml_code=extracted_code, error=None)
        else:
            return PlantUMLResponse(plantuml_code=None, error="Could not extract PlantUML code from the response.")
    raise HTTPException(status_code=500, detail="Failed to generate diagram code.")
