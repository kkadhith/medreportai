import streamlit as st
from openai import OpenAI
import PyPDF2
import os
from dotenv import load_dotenv
import faiss
import numpy as np

load_dotenv()

class MedicalReportAssistant:
    def __init__(self):
        ## ----------- Create a .env file with your OpenAI API key for this to work  ----------- 
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.index = faiss.IndexFlatL2(1536)
        self.texts = []
        
    def extractPDF(self, pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def chunkWords(self, text, chunk_size=1000):
        words = text.split()

        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def embedText(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def retrieveSummary(self):

        if not self.texts:
            return "The PDF has not been processed yet."

        full_text = ""

        for i in self.texts:
            full_text += i

        prompt = f"""You are a medical report assistant. Generate an extremely short overview of the medical report:
        {full_text}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content


    def processDocument(self, pdf_file):
        text = self.extractPDF(pdf_file)
        self.texts = self.chunkWords(text)
        
        embeddings = [self.embedText(chunk) for chunk in self.texts]
        embeddings_np = np.array(embeddings, dtype='float32')
        self.index.add(embeddings_np)
        
        return len(self.texts)
    
    def query(self, question, k=2):
        question_embedding = self.embedText(question)
        
        distances, indices = self.index.search(np.array([question_embedding], dtype='float32'), k)
        contexts = [self.texts[i] for i in indices[0]]
        

        prompt = f"""You are a medical report assistant. Use the following contexts from the medical report to answer the questions. If you cannot find the answer, say "I cannot find this information in the report." Do not answer any other questions, you are ONLY a medical report assistant.
        Contexts:
        {' '.join(contexts)}
        Question: {question}
        Answer:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
