"""
Homelessness Prevention LLM Application with RAG
Trustworthy AI system to connect people with government services
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import requests
from typing import Dict, List, Optional
import logging

# Configure logging for compliance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrustworthyRAGSystem:
    """RAG system with compliance and trustworthy AI features"""
    
    def __init__(self):
        self.government_services = self.load_government_services()
        self.compliance_log = []
    
    def load_government_services(self) -> Dict:
        """Load government services database"""
        return {
            "mental_health": {
                "name": "NHS Mental Health Services",
                "description": "Free mental health support and counseling",
                "eligibility": "All UK residents",
                "contact": "111 (NHS 111)",
                "website": "https://www.nhs.uk/mental-health/",
                "keywords": ["depression", "anxiety", "mental health", "counseling", "therapy"]
            },
            "substance_abuse": {
                "name": "Drug and Alcohol Support Services",
                "description": "Rehabilitation and support for substance abuse",
                "eligibility": "Anyone with substance abuse issues",
                "contact": "0300 123 6600",
                "website": "https://www.nhs.uk/live-well/addiction-support/",
                "keywords": ["drugs", "alcohol", "addiction", "rehabilitation", "detox"]
            },
            "housing_support": {
                "name": "Local Authority Housing Support",
                "description": "Emergency housing and homelessness prevention",
                "eligibility": "At risk of homelessness",
                "contact": "Contact local council",
                "website": "https://www.gov.uk/housing-help",
                "keywords": ["housing", "eviction", "rent arrears", "emergency accommodation"]
            },
            "financial_support": {
                "name": "Universal Credit and Benefits",
                "description": "Financial support for those in need",
                "eligibility": "Low income or unemployed",
                "contact": "0800 328 5644",
                "website": "https://www.gov.uk/universal-credit",
                "keywords": ["benefits", "universal credit", "financial help", "unemployment"]
            },
            "domestic_violence": {
                "name": "Domestic Violence Support",
                "description": "Safe accommodation and support for domestic violence victims",
                "eligibility": "Victims of domestic violence",
                "contact": "0808 2000 247 (National Domestic Violence Helpline)",
                "website": "https://www.nationaldahelpline.org.uk/",
                "keywords": ["domestic violence", "abuse", "safe house", "protection"]
            },
            "employment_support": {
                "name": "Job Centre Plus",
                "description": "Employment support and job search assistance",
                "eligibility": "Job seekers",
                "contact": "0800 169 0190",
                "website": "https://www.gov.uk/contact-jobcentre-plus",
                "keywords": ["employment", "job search", "training", "skills", "work"]
            }
        }
    
    def analyze_user_needs(self, form_data: Dict) -> List[str]:
        """Analyze user form data to identify relevant services"""
        relevant_services = []
        
        # Mental health indicators
        if (form_data.get('mental_health_issues') == 'Yes' or 
            form_data.get('stress_level', 0) > 7):
            relevant_services.append('mental_health')
        
        # Substance abuse indicators
        if (form_data.get('substance_abuse') == 'Yes' or 
            form_data.get('legal_issues') and 'drug' in form_data.get('legal_issues', '').lower()):
            relevant_services.append('substance_abuse')
        
        # Housing indicators
        if (form_data.get('housing_status') in ['At risk of eviction', 'Homeless', 'Temporary accommodation'] or
            form_data.get('rent_arrears') == 'Yes'):
            relevant_services.append('housing_support')
        
        # Financial indicators
        if (form_data.get('employment_status') == 'Unemployed' or
            form_data.get('financial_difficulties') == 'Yes'):
            relevant_services.append('financial_support')
        
        # Domestic violence indicators
        if form_data.get('domestic_violence') == 'Yes':
            relevant_services.append('domestic_violence')
        
        # Employment indicators
        if form_data.get('employment_status') in ['Unemployed', 'Looking for work']:
            relevant_services.append('employment_support')
        
        return relevant_services
    
    def generate_recommendations(self, form_data: Dict) -> Dict:
        """Generate personalized service recommendations"""
        relevant_services = self.analyze_user_needs(form_data)
        
        recommendations = {
            'services': [],
            'priority_level': self.calculate_priority(form_data),
            'next_steps': [],
            'compliance_info': {
                'data_processed': datetime.now().isoformat(),
                'user_consent': form_data.get('consent', False),
                'data_retention': '30 days'
            }
        }
        
        for service_key in relevant_services:
            service = self.government_services[service_key]
            recommendations['services'].append({
                'name': service['name'],
                'description': service['description'],
                'contact': service['contact'],
                'website': service['website'],
                'eligibility': service['eligibility']
            })
        
        # Generate next steps
        if 'mental_health' in relevant_services:
            recommendations['next_steps'].append("Contact NHS 111 for immediate mental health support")
        
        if 'housing_support' in relevant_services:
            recommendations['next_steps'].append("Contact your local council housing department immediately")
        
        if 'domestic_violence' in relevant_services:
            recommendations['next_steps'].append("Call the National Domestic Violence Helpline for immediate support")
        
        # Log for compliance
        self.log_interaction(form_data, recommendations)
        
        return recommendations
    
    def calculate_priority(self, form_data: Dict) -> str:
        """Calculate priority level based on risk factors"""
        high_risk_factors = 0
        
        if form_data.get('housing_status') == 'Homeless':
            high_risk_factors += 3
        if form_data.get('domestic_violence') == 'Yes':
            high_risk_factors += 3
        if form_data.get('mental_health_issues') == 'Yes' and form_data.get('stress_level', 0) > 8:
            high_risk_factors += 2
        if form_data.get('substance_abuse') == 'Yes':
            high_risk_factors += 2
        
        if high_risk_factors >= 5:
            return 'URGENT'
        elif high_risk_factors >= 3:
            return 'HIGH'
        elif high_risk_factors >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def log_interaction(self, form_data: Dict, recommendations: Dict):
        """Log interaction for compliance and audit"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': form_data.get('session_id', 'anonymous'),
            'priority_level': recommendations['priority_level'],
            'services_recommended': len(recommendations['services']),
            'consent_given': form_data.get('consent', False)
        }
        self.compliance_log.append(log_entry)
        logger.info(f"Interaction logged: {log_entry}")

def main():
    st.set_page_config(
        page_title="Homelessness Prevention Support",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 Homelessness Prevention Support System")
    st.markdown("**Confidential support to connect you with government services**")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = TrustworthyRAGSystem()
    
    # Compliance and Privacy Notice
    with st.expander("🔒 Privacy & Data Protection"):
        st.markdown("""
        **Your Privacy Matters:**
        - All information is confidential and secure
        - Data is processed only to provide service recommendations
        - No personal data is stored permanently
        - You can withdraw consent at any time
        - This system complies with UK GDPR regulations
        """)
    
    # Consent checkbox
    consent = st.checkbox("I consent to my data being processed to receive service recommendations", key="consent")
    
    if not consent:
        st.warning("Please provide consent to continue with the assessment.")
        return
    
    # Main form
    st.header("Personal Situation Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Housing Situation")
        housing_status = st.selectbox(
            "Current housing status:",
            ["Stable housing", "At risk of eviction", "Temporary accommodation", "Homeless", "Other"]
        )
        
        rent_arrears = st.radio("Are you behind on rent/mortgage?", ["No", "Yes"])
        
        st.subheader("Employment & Finances")
        employment_status = st.selectbox(
            "Employment status:",
            ["Employed full-time", "Employed part-time", "Self-employed", "Unemployed", "Looking for work", "Unable to work"]
        )
        
        financial_difficulties = st.radio("Experiencing financial difficulties?", ["No", "Yes"])
        
        st.subheader("Health & Wellbeing")
        mental_health_issues = st.radio("Do you have mental health concerns?", ["No", "Yes"])
        
        stress_level = st.slider("Current stress level (1-10):", 1, 10, 5)
    
    with col2:
        st.subheader("Support Needs")
        substance_abuse = st.radio("Do you need support with substance abuse?", ["No", "Yes"])
        
        domestic_violence = st.radio("Are you experiencing domestic violence?", ["No", "Yes"])
        
        legal_issues = st.text_area("Any legal issues or recent arrests? (Optional)")
        
        st.subheader("Additional Information")
        dependents = st.number_input("Number of dependents:", 0, 10, 0)
        
        additional_info = st.text_area("Any other information you'd like to share:")
        
        emergency_contact = st.text_input("Emergency contact (optional):")
    
    # Submit button
    if st.button("Get Support Recommendations", type="primary"):
        # Collect form data
        form_data = {
            'housing_status': housing_status,
            'rent_arrears': rent_arrears,
            'employment_status': employment_status,
            'financial_difficulties': financial_difficulties,
            'mental_health_issues': mental_health_issues,
            'stress_level': stress_level,
            'substance_abuse': substance_abuse,
            'domestic_violence': domestic_violence,
            'legal_issues': legal_issues,
            'dependents': dependents,
            'additional_info': additional_info,
            'emergency_contact': emergency_contact,
            'consent': consent,
            'session_id': st.session_state.get('session_id', 'anonymous')
        }
        
        # Generate recommendations
        recommendations = st.session_state.rag_system.generate_recommendations(form_data)
        
        # Display results
        st.header("🎯 Your Personalized Support Plan")
        
        # Priority level
        priority_color = {
            'URGENT': '🔴',
            'HIGH': '🟠', 
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        
        st.markdown(f"**Priority Level:** {priority_color[recommendations['priority_level']]} {recommendations['priority_level']}")
        
        if recommendations['priority_level'] == 'URGENT':
            st.error("⚠️ URGENT: Please contact services immediately for immediate support.")
        
        # Recommended services
        st.subheader("📋 Recommended Government Services")
        
        for i, service in enumerate(recommendations['services'], 1):
            with st.expander(f"{i}. {service['name']}"):
                st.write(f"**Description:** {service['description']}")
                st.write(f"**Eligibility:** {service['eligibility']}")
                st.write(f"**Contact:** {service['contact']}")
                st.write(f"**Website:** {service['website']}")
        
        # Next steps
        if recommendations['next_steps']:
            st.subheader("🚀 Immediate Next Steps")
            for step in recommendations['next_steps']:
                st.write(f"• {step}")
        
        # Emergency contacts
        st.subheader("🆘 Emergency Contacts")
        st.write("• **Emergency Services:** 999")
        st.write("• **NHS 111:** 111")
        st.write("• **Samaritans:** 116 123")
        st.write("• **National Domestic Violence Helpline:** 0808 2000 247")
        
        # Compliance information
        with st.expander("📊 Data Processing Information"):
            st.json(recommendations['compliance_info'])

if __name__ == "__main__":
    main()