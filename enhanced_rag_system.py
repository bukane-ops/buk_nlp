"""
Enhanced RAG System with Vector Database and LLM Integration
For more sophisticated government service matching
"""

import os
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import openai
from datetime import datetime
import json

class EnhancedRAGSystem:
    """Advanced RAG system with vector embeddings and LLM integration"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection(name="government_services")
        
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Initialize vector database with government services"""
        services_data = [
            {
                "id": "mental_health_nhs",
                "name": "NHS Mental Health Services",
                "description": "Comprehensive mental health support including counseling, therapy, crisis intervention, and psychiatric care. Available 24/7 through NHS 111.",
                "eligibility": "All UK residents, no referral needed for initial contact",
                "contact": "111 (NHS 111) or GP referral",
                "website": "https://www.nhs.uk/mental-health/",
                "category": "mental_health",
                "keywords": "depression anxiety mental health counseling therapy crisis intervention psychiatric care",
                "urgency_level": "high",
                "cost": "free"
            },
            {
                "id": "substance_abuse_support",
                "name": "Drug and Alcohol Support Services",
                "description": "Rehabilitation programs, detox services, counseling, and ongoing support for substance abuse issues. Includes residential and community-based treatment.",
                "eligibility": "Anyone with substance abuse issues, self-referral accepted",
                "contact": "0300 123 6600 or local addiction services",
                "website": "https://www.nhs.uk/live-well/addiction-support/",
                "category": "substance_abuse",
                "keywords": "drugs alcohol addiction rehabilitation detox counseling treatment recovery",
                "urgency_level": "high",
                "cost": "free"
            },
            {
                "id": "housing_emergency",
                "name": "Emergency Housing Support",
                "description": "Immediate housing assistance for those at risk of or experiencing homelessness. Includes emergency accommodation, housing advice, and prevention services.",
                "eligibility": "Anyone at risk of homelessness or currently homeless",
                "contact": "Local council housing department (emergency line)",
                "website": "https://www.gov.uk/housing-help",
                "category": "housing",
                "keywords": "housing homelessness emergency accommodation eviction prevention temporary housing",
                "urgency_level": "urgent",
                "cost": "free"
            },
            {
                "id": "universal_credit",
                "name": "Universal Credit and Benefits",
                "description": "Financial support for those on low income or out of work. Includes housing costs, childcare support, and disability benefits.",
                "eligibility": "UK residents with low income or unemployed",
                "contact": "0800 328 5644",
                "website": "https://www.gov.uk/universal-credit",
                "category": "financial",
                "keywords": "benefits universal credit financial support unemployment housing benefit disability",
                "urgency_level": "medium",
                "cost": "free"
            },
            {
                "id": "domestic_violence_support",
                "name": "Domestic Violence Support Services",
                "description": "Safe accommodation, legal advice, counseling, and practical support for victims of domestic violence. 24/7 helpline available.",
                "eligibility": "Victims of domestic violence and their children",
                "contact": "0808 2000 247 (National Domestic Violence Helpline)",
                "website": "https://www.nationaldahelpline.org.uk/",
                "category": "safety",
                "keywords": "domestic violence abuse safe house protection legal advice counseling",
                "urgency_level": "urgent",
                "cost": "free"
            },
            {
                "id": "employment_support",
                "name": "Job Centre Plus Employment Support",
                "description": "Job search assistance, skills training, CV help, interview preparation, and work-related benefits guidance.",
                "eligibility": "Job seekers and those wanting to improve employment prospects",
                "contact": "0800 169 0190",
                "website": "https://www.gov.uk/contact-jobcentre-plus",
                "category": "employment",
                "keywords": "employment job search training skills CV interview work benefits",
                "urgency_level": "low",
                "cost": "free"
            },
            {
                "id": "legal_aid",
                "name": "Legal Aid Services",
                "description": "Free legal advice and representation for those who cannot afford legal fees. Covers housing, family, and criminal law.",
                "eligibility": "Low income individuals facing legal issues",
                "contact": "0345 345 4 345",
                "website": "https://www.gov.uk/legal-aid",
                "category": "legal",
                "keywords": "legal aid advice representation court housing family criminal law",
                "urgency_level": "medium",
                "cost": "free"
            }
        ]
        
        # Create embeddings and store in vector database
        for service in services_data:
            # Combine text fields for embedding
            text_for_embedding = f"{service['name']} {service['description']} {service['keywords']}"
            embedding = self.embedding_model.encode(text_for_embedding).tolist()
            
            self.collection.add(
                embeddings=[embedding],
                documents=[text_for_embedding],
                metadatas=[service],
                ids=[service['id']]
            )
    
    def analyze_user_situation(self, form_data: Dict) -> str:
        """Convert form data into natural language description"""
        situation_parts = []
        
        # Housing situation
        if form_data.get('housing_status') != 'Stable housing':
            situation_parts.append(f"Housing: {form_data.get('housing_status')}")
        
        if form_data.get('rent_arrears') == 'Yes':
            situation_parts.append("Behind on rent payments")
        
        # Employment and finances
        if form_data.get('employment_status') in ['Unemployed', 'Looking for work']:
            situation_parts.append(f"Employment: {form_data.get('employment_status')}")
        
        if form_data.get('financial_difficulties') == 'Yes':
            situation_parts.append("Experiencing financial difficulties")
        
        # Health and wellbeing
        if form_data.get('mental_health_issues') == 'Yes':
            situation_parts.append("Mental health concerns")
        
        if form_data.get('stress_level', 0) > 7:
            situation_parts.append("High stress levels")
        
        # Support needs
        if form_data.get('substance_abuse') == 'Yes':
            situation_parts.append("Substance abuse issues")
        
        if form_data.get('domestic_violence') == 'Yes':
            situation_parts.append("Domestic violence situation")
        
        # Legal issues
        if form_data.get('legal_issues'):
            situation_parts.append(f"Legal issues: {form_data.get('legal_issues')}")
        
        return ". ".join(situation_parts)
    
    def find_relevant_services(self, user_situation: str, n_results: int = 5) -> List[Dict]:
        """Use vector similarity to find most relevant services"""
        # Create embedding for user situation
        situation_embedding = self.embedding_model.encode(user_situation).tolist()
        
        # Query vector database
        results = self.collection.query(
            query_embeddings=[situation_embedding],
            n_results=n_results
        )
        
        relevant_services = []
        for i, metadata in enumerate(results['metadatas'][0]):
            service_info = metadata.copy()
            service_info['similarity_score'] = 1 - results['distances'][0][i]  # Convert distance to similarity
            relevant_services.append(service_info)
        
        return relevant_services
    
    def generate_llm_recommendations(self, user_situation: str, relevant_services: List[Dict]) -> str:
        """Use LLM to generate personalized recommendations"""
        services_text = "\n".join([
            f"- {service['name']}: {service['description']} (Contact: {service['contact']})"
            for service in relevant_services
        ])
        
        prompt = f"""
        You are a compassionate social services advisor helping someone prevent homelessness.
        
        User's situation: {user_situation}
        
        Available government services:
        {services_text}
        
        Provide personalized, empathetic recommendations including:
        1. Most urgent services to contact first
        2. Step-by-step action plan
        3. What to expect from each service
        4. Any immediate safety concerns
        
        Be supportive and practical. Focus on immediate next steps.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback if OpenAI API is not available
            return self.generate_fallback_recommendations(relevant_services)
    
    def generate_fallback_recommendations(self, relevant_services: List[Dict]) -> str:
        """Generate recommendations without LLM"""
        recommendations = ["Based on your situation, here are the most relevant services:\n"]
        
        # Sort by urgency and similarity
        urgent_services = [s for s in relevant_services if s.get('urgency_level') == 'urgent']
        high_services = [s for s in relevant_services if s.get('urgency_level') == 'high']
        
        if urgent_services:
            recommendations.append("🚨 URGENT - Contact immediately:")
            for service in urgent_services:
                recommendations.append(f"• {service['name']}: {service['contact']}")
        
        if high_services:
            recommendations.append("\n🔴 HIGH PRIORITY:")
            for service in high_services:
                recommendations.append(f"• {service['name']}: {service['contact']}")
        
        return "\n".join(recommendations)
    
    def get_comprehensive_recommendations(self, form_data: Dict) -> Dict:
        """Main method to get comprehensive recommendations"""
        # Analyze user situation
        user_situation = self.analyze_user_situation(form_data)
        
        # Find relevant services using vector similarity
        relevant_services = self.find_relevant_services(user_situation)
        
        # Generate LLM recommendations
        llm_recommendations = self.generate_llm_recommendations(user_situation, relevant_services)
        
        # Calculate overall risk level
        risk_level = self.calculate_risk_level(form_data)
        
        return {
            'user_situation': user_situation,
            'relevant_services': relevant_services,
            'llm_recommendations': llm_recommendations,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'compliance_info': {
                'data_processed': True,
                'user_consent': form_data.get('consent', False),
                'retention_period': '30 days',
                'processing_purpose': 'Service recommendation'
            }
        }
    
    def calculate_risk_level(self, form_data: Dict) -> str:
        """Calculate risk level based on multiple factors"""
        risk_score = 0
        
        # Housing risk factors
        housing_risks = {
            'Homeless': 10,
            'At risk of eviction': 8,
            'Temporary accommodation': 6
        }
        risk_score += housing_risks.get(form_data.get('housing_status'), 0)
        
        # Safety risk factors
        if form_data.get('domestic_violence') == 'Yes':
            risk_score += 10
        
        # Health risk factors
        if form_data.get('mental_health_issues') == 'Yes':
            risk_score += form_data.get('stress_level', 0)
        
        if form_data.get('substance_abuse') == 'Yes':
            risk_score += 6
        
        # Financial risk factors
        if form_data.get('rent_arrears') == 'Yes':
            risk_score += 5
        
        if form_data.get('employment_status') == 'Unemployed':
            risk_score += 4
        
        # Determine risk level
        if risk_score >= 15:
            return 'CRITICAL'
        elif risk_score >= 10:
            return 'HIGH'
        elif risk_score >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'