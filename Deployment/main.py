from fastapi import FastAPI

# Create an instance of the application
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

