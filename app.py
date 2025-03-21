from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

API_KEY = "AIzaSyCM6TmUkjA24zWM4ydkxFbRtDpoNEi4qNQ"
MODEL_NAME = "gemini-2.0-flash"

genai.configure(api_key=API_KEY)

class PlantUMLRequest(BaseModel):
    prompt: str
    diagram_type: str  # Restrict to "use_case" or "architecture"
    temperature: float = 0.5  # Lower temperature for more precise output
    max_output_tokens: int = 2048

class PlantUMLResponse(BaseModel):
    plantuml_code: str | None
    error: str | None

def generate_plantuml(model_name: str, request: PlantUMLRequest) -> str | None:
    try:
        # Validate diagram_type
        if request.diagram_type not in ["use_case", "architecture"]:
            raise ValueError("diagram_type must be 'use_case' or 'architecture'")

        # Tailored prompts for each diagram type
        if request.diagram_type == "use_case":
            plantuml_prompt = f"""
            Generate PlantUML code for a Use Case Diagram based on this description:
            {request.prompt}
            - Use actors (e.g., :ActorName:) and use cases (e.g., (UseCaseName)).
            - Connect actors to use cases with -->.
            - Use concise, standard PlantUML syntax.
            - Enclose the code in @startuml and @enduml tags only.
            - Do not include explanations or additional text outside the tags.
            """
        elif request.diagram_type == "architecture":
            plantuml_prompt = f"""
            Generate PlantUML code for a System Architecture Diagram based on this description:
            {request.prompt}
            - Use components (e.g., [ComponentName]), interfaces, and nodes if specified.
            - Connect components with --> or -[hidden]-> for clarity.
            - Use concise, standard PlantUML syntax.
            - Enclose the code in @startuml and @enduml tags only.
            - Do not include explanations or additional text outside the tags.
            """

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            plantuml_prompt,
            generation_config=genai.GenerationConfig(
                temperature=request.temperature,  # Lower temperature for precision
                max_output_tokens=request.max_output_tokens,
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
    generated_text = generate_plantuml(MODEL_NAME, request)
    if generated_text:
        extracted_code = extract_plantuml_code(generated_text)
        if extracted_code:
            return PlantUMLResponse(plantuml_code=extracted_code, error=None)
        else:
            return PlantUMLResponse(plantuml_code=None, error="Could not extract PlantUML code from the response.")
    raise HTTPException(status_code=500, detail="Failed to generate diagram code.")
