from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from starlette.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from model import find_names
import numpy as np

templates = Jinja2Templates(directory="templates")

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

SQLALCHEMY_DATABASE_URL = "sqlite:///./attendancedb.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "student"
    rollnumber = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    attendance = relationship("Attendance", back_populates="student")

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String)
    rollnumber = Column(Integer, ForeignKey("student.rollnumber"))
    student = relationship("Student", back_populates="attendance")


# Pydantic models
from typing import List
from pydantic import BaseModel

class StudentBase(BaseModel):
    rollnumber: int
    name: str

class StudentCreate(StudentBase):
    pass

class StudentOut(StudentBase):
    attendance: List["AttendanceOut"] = []

    class Config:
        orm_mode = True

class AttendanceBase(BaseModel):
    id: int
    date: str
    rollnumber: int

class AttendanceCreate(AttendanceBase):
    pass

class AttendanceOut(AttendanceBase):
    student: StudentOut

    class Config:
        orm_mode = True
from sqlalchemy.orm import Session
from sqlalchemy.orm import Session
from fastapi import Depends
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session



from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# ... (previous code)
Base.metadata.create_all(bind=engine)
# Create a session
# db = SessionLocal()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recognize")
async def upload_image(request: Request, image: UploadFile, minConf: float = 0.2, network: str = 'vit_l_dp005_mask_005', weight: str = 'glint360k_model_TransFace_L.pt', db: Session = Depends(get_db)):
    try:
        # Save the uploaded image to a file
        with open("temp_image.jpg", "wb") as temp_image:
            temp_image.write(image.file.read())

        # Perform face recognition
        names = find_names("temp_image.jpg", minConf, network, weight)
        # names_list = names.split(", ")
        for roll_nos in names:
            attendance_entry1 = Attendance(date=str(datetime.today().strftime('%Y-%m-%d')), rollnumber=roll_nos)
            db.add(attendance_entry1)
            db.commit()

        # Redirect to the results page
        return templates.TemplateResponse("display.html", {"request": request, "names": names})
    except Exception as e:
        # Handle and log any errors here
        return JSONResponse(content={"error": str(e)})
    
@app.get("/students/")
def get_all_students(request: Request,db: Session = Depends(get_db)):
    students = db.query(Attendance).order_by(Attendance.rollnumber).all()
    return templates.TemplateResponse("record.html", {"request": request, "record":students })