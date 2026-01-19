import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
app = FastAPI()
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.post("/predict")
async def upload_file(file: UploadFile = File(...)):
    # Construct the path where the file will be saved
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    # Save the file to disk
    # shutil.copyfileobj is efficient for copying large files
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "saved_path": file_path}
@app.get("/")
async def main():
    return FileResponse('front.html')
