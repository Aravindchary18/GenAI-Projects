from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from services.resume_analysis_service import analyze_resume

from services.skill_gap_service import analyze_skill_gap

from services.roadmap_service import analyze_roadmap

import pdfplumber

import uuid

import os

os.makedirs("uploads", exist_ok=True)

router = APIRouter()

async def extract_pdf_text(file: UploadFile):
      
    


        if not file.filename.lower().endswith(".pdf"):        
           raise HTTPException(
            status_code=400,
            detail="only pdf's are allowed"
        )   
    
        file_path=f"uploads/{uuid.uuid4()}.pdf"
        
        try:
                
            with open(file_path,"wb") as f:
                
                while chunk:=await  file.read(1024*1024):
                
                    f.write(chunk)

            full_text =  ""

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()

                    if text:
                        full_text += text + "\n"

            if not full_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="pdf does not contain text"
                )
            return full_text

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)



        

@router.post("/career_mentor/analyze")
async def resume_analyze(file:UploadFile=File(...)):

    try:
           full_text = await extract_pdf_text(file)

           result = analyze_resume(full_text)

           return {
            "message": "Resume analyzed successfully",
            "result": result
           }
            
        
    except HTTPException:
        raise
    
    except Exception as e:
         raise HTTPException(
            status_code=500,
            detail=f"resume Upload failed: {str(e)}"
         )

    
        

       
    
@router.post("/career_mentor/skill-gap")
async def skill_gap_analysis(
    file:UploadFile=File(...),
    job_description: str=Form(...)
):
                         

    try:
        
        full_text = await  extract_pdf_text(file)

        
        
        result = analyze_skill_gap(
            full_text,
            job_description
        )

        return {
            "message": "skill gap analysis completed successfully",
            "result": result
        }
        
    except HTTPException:
        raise
    
    except Exception as e:
         raise HTTPException(
            status_code=500,
            detail=f"skill gap analysis failed: {str(e)}"
         )


@router.post("/career_mentor/roadmap")
async def career_roadmap(
    current_skills: str = Form(...),
    target_role: str = Form(...),
    experience_years: int = Form(0)

):
    try:

        skills_list = [
            skill.strip()
            for skill in current_skills.split(",")
            if skill.strip()
        ]

        result = analyze_roadmap(
            skills_list,
            target_role,
            experience_years
        )

        return {
            "message": "career roadmap generated successfully",
            "result": result
        }
        
    except HTTPException:
            raise
        
    except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"career roadmap failed: {str(e)}"
            )
        