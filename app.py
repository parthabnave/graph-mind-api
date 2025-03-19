from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

API_KEY = "AIzaSyCM6TmUkjA24zWM4ydkxFbRtDpoNEi4qNQ"
MODEL_NAME = "gemini-2.0-flash"

app = FastAPI()

def setup_gemini_api(api_key):
    genai.configure(api_key=api_key)

def generate_plantuml(model_name, prompt, temperature=0.7, top_p=1.0, top_k=1, max_output_tokens=2048):
    model = genai.GenerativeModel(model_name)
    try:
        plantuml_prompt = f"Generate PlantUML code for the following description:\n\n{prompt}\n\nPlease provide only the PlantUML code block, enclosed in @startuml and @enduml tags."

        response = model.generate_content(
            plantuml_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
            ),
        )

        return response.text
    except Exception as e:
        return None

def extract_plantuml_code(text):
    start_tag = "@startuml"
    end_tag = "@enduml"

    start_index = text.find(start_tag)
    end_index = text.find(end_tag)

    if start_index != -1 and end_index != -1:
        return text[start_index : end_index + len(end_tag)]
    else:
        return None

class PlantUMLRequest(BaseModel):
    prompt: str

class PlantUMLResponse(BaseModel):
    plantuml_code: str | None
    error: str | None

@app.post("/generate_plantuml", response_model=PlantUMLResponse)
async def generate_plantuml_api(request: PlantUMLRequest):
    setup_gemini_api(API_KEY)
    generated_plantuml = generate_plantuml(MODEL_NAME, request.prompt)

    if generated_plantuml:
        extracted_code = extract_plantuml_code(generated_plantuml)
        if extracted_code:
            return PlantUMLResponse(plantuml_code=extracted_code, error=None)
        else:
            return PlantUMLResponse(plantuml_code=None, error="Could not extract PlantUML code from the generated response.")
    else:
        raise HTTPException(status_code=500, detail="Failed to generate PlantUML code.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)