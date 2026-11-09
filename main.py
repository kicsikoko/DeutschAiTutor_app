import os
import json
import sqlite3
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Környezeti változók betöltése a .env fájlból
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Hiba: GEMINI_API_KEY nem található a .env fájlban!")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Deutsch A2 Context & Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statikus mappák csatolása a PWA / Frontendhez
app.mount("/static", StaticFiles(directory="static"), name="static")

def init_db():
    conn = sqlite3.connect("deutsch_learning.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            german_word TEXT NOT NULL,
            word_type TEXT,
            article TEXT,
            plural TEXT,
            conjugation_notes TEXT,
            hungarian_translation TEXT NOT NULL,
            example_sentences TEXT,
            grammar_explanation TEXT,
            level INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- PYDANTIC SCHEMAS ---
class WordAnalysisRequest(BaseModel):
    query: str

class WordAnalysisResponse(BaseModel):
    german_word: str
    word_type: str  # "noun", "verb", "adjective", "other"
    article: str  # "der", "die", "das", vagy üres string
    plural: str  # pl. "-e", "-en", "-¨e" vagy üres string
    conjugation_notes: str  # pl. "ruft an, hat angerufen (+Akk)"
    hungarian_translation: str
    example_sentences: List[str]  # 2-3 példamondat magyar fordítással
    grammar_explanation: str  # Rövid A2 nyelvtani magyarázat magyarul

class SaveWordRequest(BaseModel):
    german_word: str
    word_type: str
    article: str = ""
    plural: str = ""
    conjugation_notes: str = ""
    hungarian_translation: str
    example_sentences: List[str]
    grammar_explanation: str = ""

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    """Visszaadja a PWA frontend felületet (static/index.html)."""
    return FileResponse("static/index.html")

@app.post("/api/analyze", response_model=WordAnalysisResponse)
async def analyze_word(req: WordAnalysisRequest):
    """Meghívja a Gemini API-t, hogy szerkezetileg elemzzen egy német szót vagy kifejezést A2 szinten."""
    word = req.query.strip()
    if not word:
        raise HTTPException(status_code=400, detail="Üres kifejezést adtál meg!")

    prompt = f"""
    Elemzed ezt a német kifejezést vagy mondatot a magyarul tanuló A2 szintű diáknak: "{word}"
    
    Add meg a következőket:
    - german_word: A pontos német szó alapalaki formában (vagy a kifejezés).
    - word_type: "noun", "verb", "adjective" vagy "other".
    - article: Főnév esetén "der", "die", vagy "das". Egyébként üres string "".
    - plural: Főnév esetén a többes szám jele (pl. "-e", "-en", "-¨e"). Egyébként üres string "".
    - conjugation_notes: Ige esetén E/3 jelen idő, Präteritum és Perfekt alak + vonzat (pl. "ruft an, hat angerufen (+Akkusativ)"). Főnévnél/másnál üres string "".
    - hungarian_translation: Pontos magyar fordítás.
    - example_sentences: 2-3 életszerű, osztrák/német A2 mindennapi példamondat magyar fordítással zárójelben.
    - grammar_explanation: Rövid, 1-2 mondatos magyar nyelvtani magyarázat (pl. elváló igekötő, Dativ-ot vonzó prepozíció, stb.).
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WordAnalysisResponse,
                temperature=0.2,
            ),
        )
        if not response.text:
            raise HTTPException(status_code=500, detail="Üres válasz érkezett a Gemini API-tól!")

        data = json.loads(response.text)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Hiba: {str(e)}")

@app.post("/api/words/save")
def save_word(word: SaveWordRequest):
    conn = sqlite3.connect("deutsch_learning.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO vocabulary 
        (german_word, word_type, article, plural, conjugation_notes, hungarian_translation, example_sentences, grammar_explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        word.german_word,
        word.word_type,
        word.article,
        word.plural,
        word.conjugation_notes,
        word.hungarian_translation,
        json.dumps(word.example_sentences, ensure_ascii=False),
        word.grammar_explanation
    ))
    conn.commit()
    word_id = cursor.lastrowid
    conn.close()
    return {"status": "success", "id": word_id}

@app.get("/api/words")
def get_all_words():
    conn = sqlite3.connect("deutsch_learning.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, german_word, article, hungarian_translation, example_sentences, grammar_explanation FROM vocabulary ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    words = []
    for r in rows:
        words.append({
            "id": r[0],
            "german_word": r[1],
            "article": r[2],
            "hungarian_translation": r[3],
            "example_sentences": json.loads(r[4]) if r[4] else [],
            "grammar_explanation": r[5]
        })
    return words