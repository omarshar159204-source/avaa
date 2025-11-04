#!/usr/bin/env python3
"""
Real Estate Lead Automation System
V3: Streamlit Web Application
(No-Spacy Version for compatibility)
"""
import time
from datetime import datetime
import json 
from openai import OpenAI
import os
import re
import tempfile
import warnings
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import whisper
# import spacy <-- REMOVED
# from spacy.tokens import Doc <-- REMOVED
import requests
import streamlit as st
import time

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Dataclass ---
@dataclass
class FieldData:
    value: Any
    source: str
    confidence: float = 1.0

class ProcessStatus:
    def __init__(self):
        self.stages = {
            'file_upload': {'status': 'waiting', 'message': 'File Upload'},
            'data_parsing': {'status': 'waiting', 'message': 'Data Parsing'}, 
            'audio_transcription': {'status': 'waiting', 'message': 'Audio Transcription'},
            'ai_analysis': {'status': 'waiting', 'message': 'AI Analysis'},
            'qualification': {'status': 'waiting', 'message': 'Lead Qualification'},
            'report_generation': {'status': 'waiting', 'message': 'Report Generation'}
        }
    
    def update_stage(self, stage, status, message=None):
        self.stages[stage]['status'] = status
        if message:
            self.stages[stage]['message'] = message
    
    def display_status(self):
        # Display status indicators with colors
        for stage, info in self.stages.items():
            status = info['status']
            if status == 'processing': icon = '🔄'
            elif status == 'complete': icon = '✅' 
            elif status == 'warning': icon = '⚠️'
            elif status == 'failed': icon = '❌'
            else: icon = '⚪'
            
            st.write(f"{icon} {info['message']}")

# --- NLPAnalyzer Class (Spacy Removed) ---
# --- NLPAnalyzer Class (Spacy Removed) ---
class NLPAnalyzer:
    """
    Analyzes conversation transcripts using regex for motivation.
    'spacy' has been removed to ensure build compatibility.
    """
    def __init__(self):
        # self.nlp = self._load_spacy_model() <-- REMOVED
        pass # No model to load

    def analyze_transcript(self, transcript: str) -> Dict[str, str]:
        """
        Run the NLP pipeline on a transcript.
        """
        if not transcript:
            return {
                'motivation': "No transcript provided",
                'highlights': "No transcript provided"  
            }

        transcript_lower = transcript.lower()
        
        # Run individual analysis functions
        motivation = self._analyze_motivation(transcript_lower)
        
        return {
            'motivation': motivation,
            'highlights': "To be analyzed by AI"  # Placeholder for AI analysis
        }

    def _analyze_motivation(self, transcript_lower: str) -> str:
        """
        Contextually analyze motivation/timeline.
        """
        high_motivation_patterns = [
            r"asap", r"soon as possible", r"immediate(ly)?",
            r"urgent", r"quick(ly)?", r"fast", r"motivated", r"ready",
            r"i can sell it right now", r"need to sell", r"have to sell",
            r"time sensitive", r"deadline", r"quick sale", r"fast closing"
        ]
        if any(re.search(p, transcript_lower) for p in high_motivation_patterns):
            return "Highly motivated - wants quick sale"
        
        low_motivation_patterns = [
            r"flexible", r"no rush", r"whenever", r"listing it", 
            r"testing (the )?market", r"just looking", r"seeing what",
            r"not in a hurry", r"take your time", r"whenever you"
        ]
        if any(re.search(p, transcript_lower) for p in low_motivation_patterns):
            return "Flexible timeline / Testing market"
        
        # Check for moderate motivation
        moderate_patterns = [
            r"want to sell", r"looking to sell", r"interested in selling",
            r"considering offers", r"ready to move", r"planning to sell"
        ]
        if any(re.search(p, transcript_lower) for p in moderate_patterns):
            return "Moderately motivated - open to selling"
        
        return "No motivation details discussed"

# --- FormParser Class (Unchanged) ---
class FormParser:
    """Parse real estate lead forms with enhanced field processing"""
    
    def __init__(self):
        self.field_patterns = {
            'list_name': ['List Name'],
            'property_type': ['Property Type'],
            'seller_name': ['Seller Name'],
            'phone_number': ['Phone Number'],
            'address': ['Address'],
            'zillow_link': ['Zillow link'],
            'asking_price': ['Asking Price'],
            'zillow_estimate': ['Zillow Estimate'],
            'realtor_estimate': ['Realtor Estimate'],
            'redfin_estimate': ['Redfin Estimate'],
            'reason_for_selling': ['Reason For Selling'],
            'motivation_details': ['Motivation details'],
            'mortgage': ['Mortgage'],
            'condition': ['Condition'],
            'occupancy': ['Occupancy'],
            'closing_time': ['Closing time'],
            'moving_time': ['Moving time'],
            'best_time_to_call': ['Best time to call back'],
            'agent_name': ['Agent Name'],
            'call_recording': ['Call recording']
        }
    
    def parse_file(self, file_path: str) -> Dict[str, FieldData]:
        """Parse lead file into structured data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_text(content)
    
    def parse_text(self, text: str) -> Dict[str, FieldData]:
        """Extract and clean field values from text"""
        data = {}
        
        for field, names in self.field_patterns.items():
            value = self._extract_field(text, names)
            if value:
                cleaned_value = self._clean_field(field, value)
                data[field] = FieldData(value=cleaned_value, source='form')
            else:
                data[field] = FieldData(value="", source='form', confidence=0.0)
        
        return data
    
    def _extract_field(self, text: str, field_names: List[str]) -> Optional[str]:
        """Extract field value using your specific format"""
        for name in field_names:
            patterns = [
                rf'◇{name}\s*:-\s*(.+)',
                rf'◇{name}\s*:\s*(.+)',
                rf'{name}\s*:-\s*(.+)',
                rf'{name}\s*:\s*(.+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Take the last match to avoid duplicates
                    value = matches[-1].strip()
                    
                    # Special handling ONLY for ASKING PRICE to detect "not mentioned"
                    if 'asking price' in name.lower():
                        # Check for empty/not specified values for ASKING PRICE only
                        if (not value or 
                            value.lower() in ['', 'not specified', 'n/a', 'na', 'not available', 'unknown', 'not mentioned', 'not provided', 'none', 'null'] or
                            any(phrase in value.lower() for phrase in ['not mentioned', 'not provided', 'none', 'null', 'not discussed', 'no price', 'no asking'])):
                            return "Waiting for our offer"
                        
                        # Check if negotiable is mentioned in the asking price field
                        if 'negotiable' in value.lower():
                            return value + " (negotiable)"
                    
                    # For estimates (Zillow, Realtor, Redfin), always return the actual value
                    # even if it says "not specified" - we want to show what was provided
                    if value and value not in ['', 'Not specified', 'N/A', 'n/a']:
                        # Clean up malformed prefixes
                        value = re.sub(r'^◇.*Estimate\s*:-\s*', '', value)
                        value = re.sub(r'^[^a-zA-Z0-9$]*', '', value)
                        return value
        return None
    
    def _clean_field(self, field: str, value: str) -> str:
        """Clean and normalize field values with enhanced processing"""
        cleaners = {
            'phone_number': self._clean_phone,
            'seller_name': self._clean_name,
            'agent_name': self._clean_name,
            'asking_price': self._clean_price,
            'zillow_estimate': self._clean_price,
            'realtor_estimate': self._clean_price,
            'redfin_estimate': self._clean_price,
            'property_type': self._clean_property_type,
            'best_time_to_call': self._clean_time,
            'occupancy': self._clean_occupancy,
            'mortgage': self._clean_mortgage,
            'condition': self._clean_condition,
            'reason_for_selling': self._clean_reason,
            'closing_time': self._clean_closing_time,
            'moving_time': self._clean_closing_time,
            'motivation_details': self._clean_motivation,
        }
        
        cleaner = cleaners.get(field, lambda x: x.strip())
        return cleaner(value)

    
    def _clean_phone(self, phone: str) -> str:
        """Normalize phone number format"""
        digits = re.sub(r'[^\d]', '', phone)
        if len(digits) == 11 and digits.startswith('1'):
            digits = digits[1:]
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        return phone
    
    def _clean_name(self, name: str) -> str:
        return name.strip().title()

    def _clean_price(self, price: str) -> str:
        """Enhanced price cleaner that preserves negotiable notation"""
        if not price: 
            return price
        
        price = price.strip()
        
        # Check if it's already marked as negotiable
        is_negotiable = "(negotiable)" in price
        base_price = price.replace("(negotiable)", "").strip()
        
        # Only apply "Waiting for our offer" if it's explicitly set to that
        if base_price == "Waiting for our offer":
            return base_price
        
        # Remove form prefixes and clean up
        base_price = re.sub(r'^◇.*Estimate\s*:-\s*', '', base_price)
        base_price = re.sub(r'^[^a-zA-Z0-9$]*', '', base_price)
        
        # Handle "K" notation
        if base_price.upper().endswith('K'):
            try:
                numeric_value = float(base_price.upper().replace('K', '').replace('$', '').replace(',', '').strip())
                cleaned_price = f"${numeric_value * 1000:,.0f}"
                return f"{cleaned_price} (negotiable)" if is_negotiable else cleaned_price
            except ValueError: 
                pass
        
        # Extract numeric values - FIXED: Handle empty strings properly
        numbers = re.findall(r'([\d,]+\.?\d*)', base_price)
        if numbers:
            # Filter out empty strings and handle conversion safely
            valid_numbers = []
            for num_str in numbers:
                clean_num_str = num_str.replace(',', '').replace('$', '').strip()
                if clean_num_str:  # Only process non-empty strings
                    try:
                        numeric_value = float(clean_num_str)
                        valid_numbers.append((num_str, numeric_value))
                    except ValueError:
                        continue
            
            if valid_numbers:
                # Find the number with the largest magnitude
                largest_num_str, largest_value = max(valid_numbers, key=lambda x: x[1])
                # Format as integer if whole number, else keep decimals
                if largest_value == int(largest_value):
                    cleaned_price = f"${int(largest_value):,}"
                else:
                    cleaned_price = f"${largest_value:,.2f}"
                
                # Add negotiable back if it was there
                return f"{cleaned_price} (negotiable)" if is_negotiable else cleaned_price
        
        return price  # Return original if no cleaning applied
    def _clean_property_type(self, prop_type: str) -> str:
        """ V20: Simplified - just extract basic info, AI will do the cleaning """
        original = prop_type.strip()
        
        # Just do basic cleanup to remove form prefixes and excessive whitespace
        cleaned = original
        
        # Remove form prefixes
        cleaned = re.sub(r'^◇\s*Property\s*Type\s*:-\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^◇\s*Property\s*Type\s*:\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Basic cleanup
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def _clean_time(self, time: str) -> str:
        time = time.strip().lower()
        time_map = {
            'asap': 'As soon as possible',
            'anytime': 'Any time',
            'ay time': 'Any time',
        }
        return time_map.get(time, time.title())
    
    def _clean_occupancy(self, occupancy: str) -> str:
        """Enhanced occupancy cleaner that handles detailed descriptions"""
        occupancy_lower = occupancy.lower().strip()
        
        # Handle "na" and empty values
        if occupancy_lower in ['n/a', 'na', 'not available', 'unknown', 'not specified', '']:
            return "Not specified"
        
        # More specific pattern matching
        if 'vacant lot' in occupancy_lower:
            return "Vacant Lot"
        
        # Check for 30-day notice FIRST (highest priority)
        if any(phrase in occupancy_lower for phrase in [
            "30 day notice", "30-day notice", "30 days notice", 
            "notice to vacate", "submitted notice", "given notice"
        ]):
            return "Tenant Occupied (30-day notice given)"
        
        # Check for vacant patterns
        vacant_patterns = [
            'vacant', 'empty', 'no one living', 'nobody living', 
            'unoccupied', 'not occupied'
        ]
        if any(pattern in occupancy_lower for pattern in vacant_patterns):
            return "Vacant"
        
        # Check for tenant occupied patterns
        tenant_patterns = [
            'tenant', 'rented', 'renting', 'occupied by tenant', 'renter',
            'currently rented', 'has a tenant', 'tenant occupied', 'lease'
        ]
        if any(pattern in occupancy_lower for pattern in tenant_patterns):
            return "Tenant Occupied"
        
        # Check for owner occupied patterns
        owner_patterns = [
            'owner occupied', 'primary residence', 'i live here', 'we live here',
            'owner-occupied', 'living in it', 'reside there'
        ]
        if any(pattern in occupancy_lower for pattern in owner_patterns):
            return "Owner Occupied"
        
        # Return original but properly capitalized
        return occupancy.strip().capitalize()
    
    def _clean_mortgage(self, mortgage: str) -> str:
        """Enhanced mortgage cleaner that handles 'na' and other common values"""
        mortgage_lower = mortgage.lower().strip()
        
        # Handle "na", "n/a", etc.
        if mortgage_lower in ['n/a', 'na', 'not available', 'unknown', 'not specified', '']:
            return "Not available"
        
        if any(word in mortgage_lower for word in ['free and clear', 'own', 'paid off', 'no mortgage']):
            return "Owned free and clear"
        if any(word in mortgage_lower for word in ['mortgage', 'loan', 'yes', '$', 'left', 'owe']):
            return "Mortgage exists"
        return "Mortgage status unknown"
    
    def _clean_condition(self, condition: str) -> str:
        condition_lower = condition.lower()
        if 'total renovation' in condition_lower:
            return "Property requires complete renovation"
        if 'needs some repairs' in condition_lower:
            return "Property requires some repairs"
        if any(word in condition_lower for word in ['excellent', 'great', 'brand new']):
            return "Property in excellent condition"
        if any(word in condition_lower for word in ['good', 'nice', 'well maintained']):
            return "Property in good condition"
        if 'vacant lot' in condition_lower:
            return "Vacant lot"
        return condition # Return original if no clear match
    
    def _clean_reason(self, reason: str) -> str:
        reason_lower = reason.lower()
        if any(phrase in reason_lower for phrase in ['fix', 'investment', 'flip']):
            return "Property investment business"
        if "taxes" in reason_lower:
            return "Financial pressure from property taxes"
        if any(word in reason_lower for word in ['relocat', 'move']):
            return "Relocation"
        return "Standard property disposition"
    
    def _clean_closing_time(self, time: str) -> str:
        time = time.strip().lower()
        if 'asap' in time: return 'As soon as possible'
        return time.title()
    
    def _clean_motivation(self, motivation: str) -> str:
        motivation_lower = motivation.lower()
        if any(word in motivation_lower for word in ['high', 'very', 'urgent']):
            return "Highly motivated"
        if any(word in motivation_lower for word in ['motivated', 'ready']):
            return "Motivated"
        return "Standard motivation"

# --- AudioProcessor Class (with Caching) ---
class AudioProcessor:
    """Handle audio download and transcription using local Whisper"""
    
    def __init__(self):
        self.whisper_model = self._load_model()
    
    @st.cache_resource  # <-- ADDED: Cache the Whisper model
    def _load_model(_self):
        """Load Whisper model on demand"""
        with st.spinner("🧠 Loading Whisper AI model (small)... This may take a moment."):
            model = whisper.load_model("small")
        return model
    
    def transcribe_audio(self, audio_url: str, status_tracker=None) -> Dict[str, Any]:
        """Transcribe audio from URL with time estimates and failure handling"""
        
        if status_tracker:
            status_tracker.update_stage('audio_transcription', 'processing', 'Starting audio transcription...')
        
        result = {
            'transcript': None,
            'language': None, 
            'success': False,
            'error': None
        }

        # Validate URL first
        if not audio_url or 'http' not in audio_url:
            error_msg = "Invalid audio URL"
            result['error'] = error_msg
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'failed', error_msg)
            return result

        temp_path = None
        progress_placeholder = None  # <-- ADD THIS
        progress_bar = None 
        try:
            # Download audio with progress
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'processing', 'Downloading audio file...')
            
            response = requests.get(audio_url, timeout=60)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                f.write(response.content)
                temp_path = f.name
            
            # Get audio duration for time estimate
            audio_duration = self._estimate_audio_duration(temp_path)
            estimated_time = self._calculate_transcription_time(audio_duration)
            
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'processing', 
                                        f'Transcribing audio (~{estimated_time}s remaining)...')

            # Create progress indicators
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Show initial progress
            progress_bar.progress(0.1)
            progress_placeholder.text(f"Starting transcription... (~{estimated_time}s remaining)")

            # Do the actual transcription (this is the slow part)
            model = self.whisper_model
            transcription = model.transcribe(temp_path)

            # Complete the progress
            progress_bar.progress(1.0)
            progress_placeholder.text("Transcription complete!")

            # Clean up progress indicators
            progress_placeholder.empty()
            progress_bar.empty()

            if not transcription['text'].strip():
                raise Exception("Transcription returned empty content")

            result['transcript'] = transcription['text'].strip()
            result['language'] = transcription.get('language', 'en')
            result['success'] = True
            
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'complete', 'Transcription completed successfully')
            
            st.success("✅ Transcription complete")

        except Exception as e:
            error_msg = str(e)
            result['error'] = error_msg
            
            # Clear any progress indicators
            try:
                progress_placeholder.empty()
                progress_bar.empty()
            except:
                pass
                
            if status_tracker:
                status_tracker.update_stage('audio_transcription', 'failed', f'Transcription failed: {error_msg}')
            
            st.error(f"❌ Transcription failed: {error_msg}")
            
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
        return result

    def _estimate_audio_duration(self, file_path: str) -> int:
        """Estimate audio duration in seconds (simplified)"""
        try:
            file_size = os.path.getsize(file_path)
            # Rough estimate: 1MB ≈ 60 seconds of audio
            return max(30, min(600, file_size / (1024 * 1024) * 60))
        except:
            return 60  # Default fallback

    def _calculate_transcription_time(self, audio_duration: int) -> int:
        """Calculate estimated transcription time in seconds"""
        base_time = 45  # Base processing time
        return base_time + int(audio_duration * 1.5)

# --- ConversationSummarizer Class (Unchanged) ---
class ConversationSummarizer:
    """
    Generates a summary of the conversation.
    This is kept separate from NLPAnalyzer to focus on
    summary generation vs. raw data extraction.
    """
    def __init__(self):
        pass # No model needed, will receive extracted data

    def summarize(self, transcript: str, nlp_data: Dict[str, str]) -> str:
        """Generate enhanced summary based on extracted NLP data"""
        if not transcript:
            return "No transcript available for summarization."
        
        # Use the structured data from NLPAnalyzer to build a reliable summary
        key_points = []
        
        # Reason
        reason = nlp_data.get('reason', '')
        if reason and "no reason" not in reason.lower():
            key_points.append(f"Reason for Selling: {reason}")
        
        # Motivation
        motivation = nlp_data.get('motivation', '')
        if motivation and "no motivation" not in motivation.lower():
            key_points.append(f"Seller Motivation: {motivation}")
            
        # Personality
        personality = nlp_data.get('personality', '')
        if personality and "did not share" not in personality.lower():
            key_points.append(f"Seller Personality: {personality}")

        # Condition
        condition = nlp_data.get('condition', '')
        if condition and "no specific" not in condition.lower():
            key_points.append(f"Property Condition: {condition}")
            
        # Mortgage
        mortgage = nlp_data.get('mortgage', '')
        if mortgage and "no mortgage" not in mortgage.lower():
            key_points.append(f"Mortgage Status: {mortgage}")

        # Occupancy
        occupancy = nlp_data.get('tenant', '')
        if occupancy and "no occupancy" not in occupancy.lower():
            key_points.append(f"Occupancy: {occupancy}")
        
        
        summary_lines = [
            "ENHANCED CONVERSATION ANALYSIS",
            "=" * 60,
            "",
            "KEY DISCUSSION POINTS:",
            "-" * 40
        ]
        
        if key_points:
            for point in key_points:
                summary_lines.append(f"• {point}")
        else:
            summary_lines.append("• Key details were not clearly discussed in the conversation.")
            
        summary_lines.append("")
        return "\n".join(summary_lines)

# --- AIRephraser Class (with Caching) ---
# --- AIRephraser Class (with Caching) ---
class AIRephraser:
    """
    V11: Complete AI analysis for all major conversation topics
    Handles: Reason, Condition, Mortgage, Occupancy
    """
    
    def __init__(self):
        """
        Initialize the AI Rephraser with the DeepSeek API client.
        """
        self.client = None
        self.model = "deepseek-chat"
        self._initialize_client()
    
    def _initialize_client(self):
        """
        Initialize the API client with caching.
        """
        self.client = self._get_cached_client()
    
    @st.cache_resource
    def _get_cached_client(_self):
        """
        Cached method to get the API client.
        """
        try:
            # Get key from Streamlit secrets
            api_key = st.secrets["DEEPSEEK_API_KEY"]

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            return client
        
        except Exception as e:
            st.error(f"❌ Failed to initialize DeepSeek client: {e}")
            return None

    def rephrase(self, topic_name: str, transcript: str) -> str:
        """
        Analyzes transcript using the DeepSeek API for a specific topic.
        """
        if not self.client:
            return f"DeepSeek API client not initialized. Cannot analyze {topic_name}."
            
        if not transcript or len(transcript) < 2: # Lowered threshold for short inputs like "N/A"
            return "Transcript too short for analysis."

        question = ""
        system_prompt = ""

        if topic_name == "Reason for Selling":
            # --- REASON FOR SELLING PROMPT ---
            system_prompt = f"""
            You are an expert real estate call analyst.
            Your job is to analyze the following call transcript and rephrase
            the seller's stated reason for selling into a single, complete sentence.
            
            CRITICAL INSTRUCTIONS:
            - Write it as a narrative statement (e.g., "The seller is moving to...")
            - DO NOT use bullet points.
            - DO NOT add "The seller stated:". Just write the sentence.
            - EXAMPLE: "He is going to live with his father to take care of him because he has cancer."
            - If no reason is mentioned, say "No reason discussed in conversation".
            - Use the seller's own words as much as possible.
            - Be specific and concise.

            Transcript:
            {transcript}
            """
            question = "What is the seller's stated reason for selling the property?"

        elif topic_name == "Property Condition":
            # --- PROPERTY CONDITION PROMPT ---
            system_prompt = f"""
            You are an expert real estate call analyst.
            Your job is to analyze the following call transcript and summarize
            the seller's description of the property's condition into a narrative paragraph.
            
            CRITICAL INSTRUCTIONS:
            - Combine all details (repairs, roof, updates, etc.) into one flowing statement.
            - DO NOT use bullet points.
            - DO NOT add "The seller stated:". Just write the paragraph.
            - EXAMPLE: "The property is in good shape but needs a new roof which will cost $9,500. The interior has been well-maintained and the house is ready for immediate occupancy."
            - If no condition is mentioned, say "No specific condition details discussed".
            - Focus on key issues: roof, HVAC, foundation, cosmetic updates, major repairs.
            - Mention specific costs if discussed.

            Transcript:
            {transcript}
            """
            question = "What is the seller's description of the property's condition?"

        elif topic_name == "Mortgage Status":
            # --- MORTGAGE STATUS PROMPT ---
            system_prompt = f"""
            You are an expert real estate call analyst.
            Analyze this conversation transcript and determine the mortgage status of the property.
            
            CRITICAL INSTRUCTIONS:
            - Be VERY specific about mortgage status
            - put in consider the input value of mortage of the form data , compare it with the transcript and give your final answer 
            - If mortgage exists, mention any amounts discussed
            - If owned free and clear, state that clearly
            - If no mortgage information is discussed, say "No mortgage information discussed"
            - Use clear, direct language
            -take in consideration that the whisper may have some errors , so be careful while giving the final answer
            - Examples of good responses:
              * "Owned free and clear - no mortgage"
              * "Mortgage exists - owes approximately $50,000"
              * "Mortgage exists but amount not specified"
              * "No mortgage information discussed in the conversation"
              - dont make the answer to large , keep it simple ,ex : owned and free clear , mortgage exists

            Transcript:
            {transcript}
            """
            question = "What is the mortgage status of the property based on the conversation?"

        elif topic_name == "Occupancy Status":
            # --- OCCUPANCY STATUS PROMPT ---
            system_prompt = f"""
            You are an expert real estate call analyst.
            Analyze this conversation transcript and determine the occupancy status of the property.
            
            CRITICAL INSTRUCTIONS:
            - Be VERY specific about occupancy status
            - Options: "Owner Occupied", "Tenant Occupied", "Vacant", "Vacant Lot"
            - If tenant occupied with notice period, specify: "Tenant Occupied (30-day notice given)"
            - If no occupancy information is discussed, say "No occupancy information discussed"
            - Use clear, direct language
            - Examples of good responses:
              * "Tenant Occupied"
              * "Owner Occupied" 
              * "Vacant"
              * "Tenant Occupied (30-day notice given)"
              * "Vacant Lot"
              * "No occupancy information discussed"
              - dont give reason for occupancy just give the status

            Transcript:
            {transcript}
            """
            question = "What is the occupancy status of this property based on the conversation?"

        elif topic_name == "Seller Personality":
            # --- SELLER PERSONALITY PROMPT ---
            system_prompt = f"""
            You are an expert real estate call analyst.
            Analyze this conversation transcript to understand the seller's personality and communication style.
            
            CRITICAL INSTRUCTIONS:
            - Summarize their communication style (e.g., "Friendly and talkative," "Strictly business," "Seems stressed and in a hurry," "Calm and patient," "Sounds elderly and a bit confused").
            - Note any personal details they *voluntarily* shared that provide context (e.g., "Mentioned a new job," "Spoke about his father being sick," "Complained about tenants").
            - DO NOT make assumptions or state information that isn't in the transcript.
            - Combine this into a short, 1-2 sentence summary.
            - If no clear personality details are available, say "Seller was professional and did not share personal details."

            Transcript:
            {transcript}
            """
            question = "Summarize the seller's personality and communication style based on the transcript."
            
        # In the AIRephraser class, update the Property Type section:
        elif topic_name == "Property Type":
            # --- (MODIFIED) PROPERTY TYPE CLEANING PROMPT ---
            system_prompt = f"""
            You are an expert real estate data formatter. Your job is to clean and standardize a raw property type description from a form.

            ### CRITICAL FORMATTING RULES ###
            1.  **Final Format:** "PropertyType (X unit), Y Bedrooms, Z Bathrooms, SQFT Square Feet"
            2.  **Commas are required:** Use commas to separate all elements.
            3.  **No Slashes:** DO NOT use slashes (/).
            4.  **No Abbreviations:** DO NOT use abbreviations like 'bed', 'beds', 'ba', 'sf', 'sqft'. Always write out "Bedrooms", "Bathrooms", "Square Feet".
            5.  **Capitalization:** Capitalize property types (e.g., "Single Family", "Duplex", "MultiFamily").
            6.  **Units:** Always include a unit count in parentheses, e.g., "(1 unit)", "(2 unit)".
            7.  **Plurals:** Use plurals correctly: "1 Bedroom", "2 Bedrooms", "1 Bathroom", "2.5 Bathrooms".
            8.  **Vacant Land:** If it is land, the format is "Vacant land, X acres".

            ### GOOD EXAMPLES ###
            * Input: "Single-family home (3 bed, 1 bath, ~1,344 sq ft)"
            * Output: "Single Family (1 unit), 3 Bedrooms, 1 Bathroom, 1,344 Square Feet"
            
            * Input: "MultiFamily (2 units), 3 bedrooms and 2 bathrooms, 1,344 sqft"
            * Output: "MultiFamily (2 unit), 3 Bedrooms, 2 Bathrooms, 1,344 Square Feet"
            
            * Input: "4-plex (4 units), 2 beds 1 bath each, 850 sqft"
            * Output: "4-plex (4 unit), 2 Bedrooms, 1 Bathroom, 850 Square Feet each"
            
            * Input: "vacant land, 0.5 acres"
            * Output: "Vacant land, 0.5 acres"
            
            * Input: "Not specified" or "N/A"
            * Output: "Not specified in form"

            ### BAD EXAMPLES (WHAT TO AVOID) ###
            * Input: "SingleFamily/3beds/1bath"
            * WRONG Output: "SingleFamily/3beds/1bath"
            * CORRECT Output: "Single Family (1 unit), 3 Bedrooms, 1 Bathroom"
            
            * Input: "sfh 3/1"
            * WRONG Output: "sfh 3/1"
            * CORRECT Output: "Single Family (1 unit), 3 Bedrooms, 1 Bathroom"

            You must always reformat the input. Never return the raw input.
            Always follow the format exactly.
            make sure everything is correct , the number of the bedrooms and bathrooms are correct
            please make sure of it before giving the final answer
            
            Property type from form to clean:
            {transcript}
            """
            question = "Clean and format this property type description from the form according to the rules."
        # Add this new condition in the rephrase method of AIRephraser class:
        elif topic_name == "Important Highlights":
            # --- IMPORTANT CALL HIGHLIGHTS PROMPT ---
            system_prompt = f"""
            You are an expert real estate call analyst.
            Analyze this conversation transcript and identify the 3-5 most important highlights that a real estate investor should know.
            
            FOCUS ON:
            - Urgent motivations or timelines
            - Financial pressures (taxes, mortgage, repairs needed)
            - Property condition issues or recent updates
            - Unique selling circumstances (inheritance, divorce, relocation)
            - Willingness to negotiate or flexibility
            - Any red flags or opportunities
            
            FORMATTING RULES:
            - Return as bullet points (use • character)
            - Maximum 5 bullet points
            - Each bullet should be 1-2 sentences max
            - Be specific and actionable
            - Focus on what matters for investment decisions
            
            Examples of good highlights:
            • Seller urgently needs to sell within 30 days due to job relocation
            • Property needs new roof - seller mentioned $9,500 repair cost
            • Seller inherited property and wants quick cash sale
            • Willing to accept 15% below asking price for fast closing
            • Tenant occupied with lease ending in 60 days
            
            If no important highlights are found, return: "No critical highlights identified in the conversation."
            
            Transcript:
            {transcript}
            """
            question = "What are the most important highlights from this conversation that a real estate investor should know?"

        else:
            return f"No analysis defined for topic: {topic_name}"

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            answer = chat_completion.choices[0].message.content.strip()
            
            # Post-processing check for the property type
            if topic_name == "Property Type":
                # If AI fails and returns slashes, return a safe default
                if '/' in answer or 'beds' in answer.lower():
                    # Return the original messy input so the user sees it,
                    # instead of a confusing error message.
                    return transcript 
            
            return answer

        except Exception as e:
            st.error(f"❌ DEEPSEEK API ERROR: {e}")
            return f"Error analyzing {topic_name} with API."
            
class AIQualifier:
    """
    Analyzes final lead data against a set of business rules
    by calling the DeepSeek AI for a final qualification.
    """
    def __init__(self, client):
        """
        Initialize the qualifier with a shared API client.
        """
        self.client = client
        self.model = "deepseek-chat"
        self.re = re # For parsing prices in the fallback

    def _get_fallback_results(self, error_msg: str) -> Dict[str, Any]:
        """Returns a standard error dictionary if the AI fails."""
        return {
            'total_score': 0,
            'verdict': "ERROR",
            'breakdown': {
                'price': {'score': 0, 'notes': f"AI qualification failed: {error_msg}"},
                'reason': {'score': 0, 'notes': f"AI qualification failed: {error_msg}"},
                'closing': {'score': 0, 'notes': f"AI qualification failed: {error_msg}"},
                'condition': {'score': 0, 'notes': f"AI qualification failed: {error_msg}"},
            }
        }

    def _get_val(self, data: Dict[str, FieldData], key: str) -> str:
        """Helper to safely get a value from the lead data dict."""
        return data.get(key, FieldData("Not Provided", "")).value

    def qualify(self, lead_data: Dict[str, FieldData]) -> Dict[str, Any]:
        """
        Runs the AI-powered qualification logic on the final, merged lead data.
        """
        if not self.client:
            return self._get_fallback_results("API client not initialized")

        # --- 1. Format the data for the AI prompt ---
        try:
            data_summary = f"""
            LEAD DATA:
            - Asking Price: {self._get_val(lead_data, 'asking_price')}
            - Zillow Estimate: {self._get_val(lead_data, 'zillow_estimate')}
            - Realtor Estimate: {self._get_val(lead_data, 'realtor_estimate')}
            - Redfin Estimate: {self._get_val(lead_data, 'redfin_estimate')}
            - Reason for Selling: {self._get_val(lead_data, 'reason_for_selling')}
            - Closing Time: {self._get_val(lead_data, 'closing_time')}
            - Property Condition: {self._get_val(lead_data, 'condition')}
            """
        except Exception as e:
            return self._get_fallback_results(f"Failed to format data: {e}")

        # --- 2. Create the AI system prompt ---
        system_prompt = f"""
        You are an expert real estate lead qualification analyst. Your job is to analyze the following lead data and score it according to a strict set of rules.

        QUALIFICATION RULES:
        1.  **Reason for Selling (50 points):**
            -   The reason MUST be a "solid reason" (e.g., relocation, divorce, financial trouble, inheritance, major life event).
            -   "Weak reasons" (e.g., "don't need it anymore," "standard disposition," "no reason discussed," "not specified") get 0 points.
            -   **Award 50 points for a solid reason, 0 for a weak one.**

        2.  **Asking Price (20 points):**
            -   First, calculate the average of all available market estimates (Zillow, Realtor, Redfin).
            -   Then, check if the "Asking Price" is *below* that average market value.
            -   If no asking price or no market estimates are provided, this fails.
            -   **Award 20 points if it's below market, 0 otherwise.**

        3.  **Closing Time (20 points):**
            -   The "Closing Time" must be 6 months or less (e.g., "ASAP," "30 days," "flexible," "6 months").
            -   If the time is over 6 months (e.g., "7 months," "next year") or not provided, it fails.
            -   **Award 20 points if it's <= 6 months, 0 otherwise.**

        4.  **Property Condition (10 points):**
            -   Any specific details about the condition must be provided.
            -   If the condition is "not specified," "no details discussed," or "no transcript," it fails.
            -   **Award 10 points if *any* condition details are present, 0 otherwise.**

        TASK:
        Analyze this data blob, calculate the score for each rule, and provide a total score and verdict.

        LEAD DATA TO ANALYZE:
        {data_summary}

        FINAL INSTRUCTIONS:
        -   You MUST return your answer in a valid JSON format.
        -   The JSON MUST match this exact structure:
        {{
          "total_score": <number>,
          "verdict": "<string: 'PRIME LEAD' (>=80 pts), 'Review' (50-79 pts), or 'REJECT' (<50 pts)>",
          "breakdown": {{
            "price": {{ "score": <number>, "notes": "<string: Your brief justification>" }},
            "reason": {{ "score": <number>, "notes": "<string: Your brief justification>" }},
            "closing": {{ "score": <number>, "notes": "<string: Your brief justification>" }},
            "condition": {{ "score": <number>, "notes": "<string: Your brief justification>" }}
          }}
        }}
        -   Be strict with the rules.
        -   Do not include any text outside the JSON.
        """

        # --- 3. Call the API and parse the JSON response ---
        try:
            with st.spinner("⚖️ Calling DeepSeek AI for final qualification..."):
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Analyze the lead data and return the qualification JSON."}
                    ],
                    temperature=0.0, # We want deterministic, rule-based output
                    max_tokens=500,
                    response_format={"type": "json_object"} # Ask for JSON output
                )
            
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Parse the JSON
            results = json.loads(response_text)
            
            # Simple validation to ensure the structure is correct
            if 'total_score' not in results or 'breakdown' not in results:
                raise ValueError("AI response missing required keys")
                
            st.success(f"⭐ AI Lead Score: {results['total_score']}/100 ({results['verdict']})")
            return results

        except json.JSONDecodeError as e:
            st.error(f"❌ AI QUALIFICATION ERROR: Failed to decode JSON: {e}")
            st.error(f"   Raw AI Response: {response_text}")
            return self._get_fallback_results(f"AI returned invalid JSON: {e}")
        except Exception as e:
            st.error(f"❌ AI QUALIFICATION ERROR: {e}")
            return self._get_fallback_results(str(e))

# --- DataValidator Class (Unchanged) ---
class DataValidator:
    """Validate and resolve data contradictions"""
    
    def clean_reason_field(self, reason_text: str, max_length: int = 500) -> str:
        """Prevent over-population of reason field"""
        if not reason_text or reason_text.strip() == "" or reason_text == "None":
            return "No reason discussed in conversation"
        
        # If it's one of our "default" messages, just return it
        if any(phrase in reason_text.lower() for phrase in ['no reason', 'did not discuss', 'not specified']):
            return reason_text
        
        # Truncate if needed
        if len(reason_text) > max_length:
            return reason_text[:max_length-3] + "..."
        
        return reason_text.strip()
    
    def validate_price_data(self, form_price: str, market_estimates: Dict[str, str]) -> Dict[str, Any]:
        """Validate price data against market estimates"""
        # ... (This logic from your original script is good and remains unchanged) ...
        return {'value': form_price, 'is_realistic': True, 'notes': []} # Placeholder
        
    def _extract_numeric_price(self, price_str: str) -> Optional[float]:
        """Extract numeric value from price string"""
        # ... (This logic from your original script is good and remains unchanged) ...
        return None # Placeholder

# --- DataMerger Class (Unchanged) ---
class DataMerger:
    """Intelligently merge form data with conversation insights"""
    
    def merge(self, form_data: Dict[str, FieldData], transcript: str, audio_analysis: Dict[str, Any]) -> Dict[str, FieldData]:
        """Merge data with intelligent field completion"""
        merged = form_data.copy()
        
        # FIX MOVING TIME FOR VACANT LOTS
        moving_time_data = merged.get('moving_time')
        moving_time_value = moving_time_data.value if moving_time_data else ""

        if not moving_time_value or moving_time_value == "Not mentioned":
            property_type_data = merged.get('property_type')
            property_type = property_type_data.value if property_type_data else ""
            
            if 'vacant lot' in str(property_type).lower():
                merged['moving_time'] = FieldData(
                    value="Not applicable - vacant lot",
                    source='derived',
                    confidence=0.9
                )
            else:
                closing_time_data = merged.get('closing_time')
                closing_time = closing_time_data.value if closing_time_data else ""
                
                if closing_time and closing_time != "Not mentioned" and closing_time != "":
                    merged['moving_time'] = FieldData(
                        value=closing_time,
                        source='derived',
                        confidence=0.7
                    )
        
        return merged

# --- ReportGenerator Class (Spacy Removed) ---
class ReportGenerator:
    """
    Generate professional text reports with enhanced data.
    V4: Form data first, in a single block as specified by user.
    """
    
    def __init__(self):
        self.conversation_summarizer = ConversationSummarizer()
        # self.nlp = None  <-- REMOVED
        
        # Define a single master list for all form fields in the user's specified order
        self.form_fields = [
            ('list_name', 'List Name'),
            ('property_type', 'Property Type'),
            ('seller_name', 'Seller Name'),
            ('phone_number', 'Phone Number'),
            ('address', 'Address'),
            ('zillow_link', 'Zillow link'),
            ('asking_price', 'Asking Price'),
            ('zillow_estimate', 'Zillow Estimate'),
            ('realtor_estimate', 'Realtor Estimate'),
            ('redfin_estimate', 'Redfin Estimate'),
            ('reason_for_selling', 'Reason For Selling'),
            ('mortgage', 'Mortgage'),
            ('condition', 'Condition'),
            ('occupancy', 'Occupancy'),
            ('closing_time', 'Closing time'),
            ('moving_time', 'Moving time'),
            ('best_time_to_call', 'Best time to call back'),
            ('agent_name', 'Agent Name'),
            ('call_recording', 'Call recording'),
        ]


        # In the ReportGenerator class, update the ai_analysis_fields:
        self.ai_analysis_fields = [
            ('personality', 'Seller Personality'),
            ('highlights', 'Important Call Highlights'),  # Add this line
        ]

        # Call recording will be in its own section at the end
        self.call_data_fields = [
            ('call_recording', 'Call recording')
        ]


    # set_nlp_model method <-- REMOVED

    def _format_field_line(self, data: Dict[str, FieldData], field_key: str, display_name: str) -> str:
        """Helper to format a single ◇ line."""
        if field_key in data:
            field_data = data[field_key]
            value = field_data.value

            if value and value != "Not mentioned" and value != "":
                # Truncate long fields
                if len(str(value)) > 400:
                    value = str(value)[:397] + "..."
                return f"◇{display_name}: {value}"
            else:
                return f"◇{display_name}: Not mentioned"
        else:
            return f"◇{display_name}: Not mentioned"

    def _format_section(self, title: str, fields_list: List[tuple], data: Dict[str, FieldData]) -> List[str]:
        """Formats a logical section with a title and fields."""
        lines = [
            "-" * 50,
            f"{title.upper()}",
            "-" * 50,
            ""
        ]
        for field_key, display_name in fields_list:
            lines.append(self._format_field_line(data, field_key, display_name))
        
        lines.append("") # Add spacing after the section
        return lines

    def _format_qualification_section(self, results: Dict[str, Any]) -> List[str]:
        """Formats the AI lead qualification score into a readable section."""
        lines = [
            "=" * 50,
            f"LEAD QUALIFICATION: {results['verdict']} ({results['total_score']} / 100)",
            "=" * 50,
            ""
        ]
        
        # Helper to format each line
        def format_score(name, data_key):
            data = results['breakdown'].get(data_key, {'score': 0, 'notes': 'N/A'})
            return f"• {name}: {data['score']} pts"

        lines.append(format_score("Reason for Selling (50%)", "reason"))
        lines.append(format_score("Asking Price (20%)", "price"))
        lines.append(format_score("Closing Time (20%)", "closing"))
        lines.append(format_score("Property Condition (10%)", "condition"))
        
        # Add a more detailed notes section
        lines.append("\nQUALIFICATION NOTES:")
        lines.append(f"- REASON: {results['breakdown'].get('reason', {}).get('notes', 'N/A')}")
        lines.append(f"- PRICE: {results['breakdown'].get('price', {}).get('notes', 'N/A')}")
        lines.append(f"- CLOSING: {results['breakdown'].get('closing', {}).get('notes', 'N/A')}")
        lines.append(f"- CONDITION: {results['breakdown'].get('condition', {}).get('notes', 'N/A')}")
        
        return lines + [""]

    def generate_report(self, merged_data: Dict[str, FieldData], 
                        transcript: Optional[str],
                        audio_result: Dict[str, Any],
                        nlp_data: Dict[str, str], # This now contains 'personality'
                        qualification_results: Dict[str, Any],
                        source_filename: str) -> str:
        """Generate enhanced text report with Form Data First in a single block"""
        
        lines = [
            "ENHANCED REAL ESTATE PROPERTY REPORT",
            "=" * 50,
            f"Source File: {os.path.basename(source_filename)}",
            ""
        ]

        # --- 1. THE FORM (Main Data Block) ---
        lines.extend(self._format_section("PROPERTY & SELLER DETAILS", self.form_fields, merged_data))
        
        # --- 2. AI CONVERSATION ANALYSIS (Personality Section) ---
        # Create a temporary data dict for AI analysis fields
        ai_data = {}
        if 'personality' in nlp_data:
            ai_data['personality'] = FieldData(
                value=nlp_data['personality'],
                source='conversation',
                confidence=0.9
            )
        
        # Only add AI Analysis section if we have personality data
        if ai_data:
            lines.extend(self._format_section("AI CONVERSATION ANALYSIS", self.ai_analysis_fields, ai_data))
        
        # --- 3. QUALIFICATION (Now after form and AI data) ---
        lines.extend(self._format_qualification_section(qualification_results))

        # --- 4. FULL TRANSCRIPT ---
        if transcript:
            lines.extend(self._format_full_transcript(merged_data, transcript))
        else:
            lines.extend([
                "-" * 50,
                "FULL CALL TRANSCRIPT",
                "-" * 50,
                "",
                "No call recording available for analysis.",
                ""
            ])
        
        # --- 5. CALL & SOURCE DATA ---
        lines.extend(self._format_section("CALL & SOURCE DATA", self.call_data_fields, merged_data))
        
        return "\n".join(lines)
    
    
    def _format_full_transcript(self, data: Dict[str, FieldData], transcript: str) -> List[str]:
        """
        Formats the transcript with heuristic speaker labels (Agent/Seller).
        This version splits by newline, not spacy sentences.
        """
        lines = [
            "-" * 50,
            "FULL CALL TRANSCRIPT",
            "-" * 50,
            "(Note: Speaker labels are a 'best guess' and only added where confident.)",
            ""
        ]

        # Get speaker names from the data, with defaults
        agent_name_full = data.get('agent_name', FieldData("Agent", "", 0.0)).value
        seller_name_full = data.get('seller_name', FieldData("Seller", "", 0.0)).value

        # Use first names as labels
        agent_label = agent_name_full.split()[0].strip(":") if agent_name_full else "Agent"
        seller_label = seller_name_full.split()[0].strip(":") if seller_name_full else "Seller"

        # Set max label length for clean formatting
        max_label_len = max(len(agent_label), len(seller_label)) + 1

        # --- MODIFIED: Loop over split lines instead of doc.sents ---
        for text in transcript.splitlines():
            text = text.strip()
            if not text:
                continue

            text_lower = text.lower()
            label = "" # Start with no label

            # --- Heuristic Rules ---
            # 1. Agent identifies themself
            if (("this is " + agent_label.lower()) in text_lower or \
                ("my name is " + agent_label.lower()) in text_lower):
                label = agent_label

            # 2. Seller says "speaking"
            elif "speaking" in text_lower and len(text_lower) < 20:
                label = seller_label

            # 3. Seller identifies themself
            elif (("this is " + seller_label.lower()) in text_lower or \
                ("my name is " + seller_label.lower()) in text_lower):
                label = seller_label

            # --- Format the line ---
            if label:
                # Add colon and pad for alignment
                formatted_label = (label + ":").ljust(max_label_len)
                lines.append(f"{formatted_label} {text}")
            else:
                # No confident label, just indent the text
                formatted_label = " ".ljust(max_label_len + 1)
                lines.append(f"{formatted_label} {text}")

        return lines + [""]
    
    def save_report(self, report_content: str, source_filename: str, 
                    output_dir: str = "output") -> str:
        """Save report to file"""
        # We don't need to save to disk in Streamlit, we'll return the content
        # But we'll create a temp file path for the name
        base_name = os.path.splitext(os.path.basename(source_filename))[0]
        output_path = f"{base_name}_enhanced_report.txt"
        
        return output_path # Just return the name

# --- RealEstateAutomationSystem Class (Spacy Removed) ---
class RealEstateAutomationSystem:
    def __init__(self):
        self.form_parser = FormParser()
        self.audio_processor = AudioProcessor()
        self.data_merger = DataMerger()
        self.report_generator = ReportGenerator() 
        self.data_validator = DataValidator()
        self.ai_qualifier = None

        # Initialize placeholders for analyzers
        self.nlp_analyzer = None
        self.rephraser = None

        self._initialize_analyzers()


    def _initialize_analyzers(self):
        """
        Initialize analyzers fresh for each lead.
        - Load spaCy once and share it.
        - Load Transformers QA model.
        """
        st.info("Loading AI models (this is cached and only runs once)...")
        
        # NLPAnalyzer loads (but doesn't hold a model)
        self.nlp_analyzer = NLPAnalyzer() # <-- MODIFIED
        
        # Initialize the new API-based rephraser (cached)
        self.rephraser = AIRephraser()

        if self.rephraser.client:
            self.ai_qualifier = AIQualifier(client=self.rephraser.client)
        else:
            st.error("❌ AI Qualifier NOT initialized (API client missing).")

        # No spacy model to set
        # self.report_generator.set_nlp_model(spacy_model) <-- REMOVED

    def process_multi_property_lead(self, input_file_path: str, input_filename: str) -> tuple[str, str]:
        """
        Process multi-property leads using AI to generate complete analysis.
        This is SEPARATE from the single-property pipeline.
        """
        # Step 1: Read the original lead data
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_lead_data = f.read()
        
        # Step 2: Parse basic form data to get call recording URL
        st.info("📝 Parsing basic form data for multi-property lead...")
        form_data = self.form_parser.parse_file(input_file_path)
        
        call_recording_url = form_data.get('call_recording').value if form_data.get('call_recording') else None
        
        # Step 3: Process audio transcription if available
        transcript = None
        if call_recording_url and call_recording_url.strip():
            st.info("🎵 Processing call recording for multi-property analysis...")
            audio_result = self.audio_processor.transcribe_audio(call_recording_url)
            
            if audio_result['success']:
                transcript = audio_result['transcript']
                st.success("✅ Transcription complete for multi-property analysis")
            else:
                st.error(f"❌ Audio processing failed: {audio_result.get('error')}")
                transcript = "No transcription available"
        else:
            st.warning("⚠️ No call recording URL found. Using form data only.")
            transcript = "No call recording available"
        
        # Step 4: Use AI to generate complete multi-property analysis
        st.info("🧠 Generating comprehensive multi-property analysis with AI...")
        report_content = self._generate_multi_property_analysis(original_lead_data, transcript)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        output_filename = f"{base_name}_multi_property_enhanced_report.txt"
        
        return report_content, output_filename

    def _generate_multi_property_analysis(self, lead_data: str, transcript: str) -> str:
        """
        Use AI to generate a complete analysis for multi-property leads.
        """
        if not self.rephraser.client:
            return "Error: AI client not available for multi-property analysis."
        
        system_prompt = """
        You are an expert real estate analyst specializing in multi-property portfolio analysis.
        Your task is to analyze a lead that contains multiple properties and generate a comprehensive,
        professional real estate report.

        CRITICAL FORMATTING RULES:
        1. Use EXACTLY the same field structure as single property reports
        2. Format: "◇Field Name: value" for each field
        3. Include ALL standard fields from the single property template
        4. For multi-property fields (Address, Property Type, Zillow links, Estimates):
        - Use "first:", "second:", "third:" format
        - Keep all original information
        - Add AI insights where helpful

        ANALYSIS GUIDELINES:
        - Extract and organize all property details clearly
        - Calculate combined value analysis
        - Identify portfolio synergies and advantages
        - Provide strategic insights for the entire package
        - Maintain professional real estate reporting standards

        EXAMPLE OUTPUT STRUCTURE:
        ENHANCED REAL ESTATE PROPERTY REPORT
        ==================================================
        Source File: [filename]

        PROPERTY & SELLER DETAILS
        --------------------------------------------------
        ◇List Name: [value]
        ◇Property Type: 
        first: [detailed description]
        second: [detailed description] 
        third: [detailed description]
        ◇Seller Name: [value]
        ◇Phone Number: [value]
        ◇Address:
        first: [full address]
        second: [full address]
        third: [full address]
        [continue with all other fields...]

        Make the report comprehensive yet concise, highlighting the unique aspects of this multi-property opportunity.
        """

        user_prompt = f"""
        LEAD FORM DATA:
        {lead_data}

        CALL TRANSCRIPT (for additional context):
        {transcript if transcript else "No transcript available"}

        Please analyze this multi-property lead and generate a complete enhanced real estate report 
        following the exact formatting guidelines. Include all original information plus your professional 
        analysis and insights about this property portfolio.
        """

        try:
            with st.spinner("🤖 AI generating multi-property analysis..."):
                chat_completion = self.rephraser.client.chat.completions.create(
                    model=self.rephraser.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
            
            report_content = chat_completion.choices[0].message.content.strip()
            st.success("✅ Multi-property analysis complete!")
            return report_content

        except Exception as e:
            st.error(f"❌ Multi-property AI analysis failed: {e}")
            return f"Error generating multi-property analysis: {e}"

    
    def process_single_property_lead(self, input_file_path: str, input_filename: str) -> tuple[str, str]:
        """Process a single lead file and return report content and filename"""
        
         # Initialize status tracker
        status_tracker = ProcessStatus()
        status_tracker.update_stage('file_upload', 'complete')
        
        # Display initial status
        st.subheader("🔄 Processing Status")
        status_tracker.display_status()
        # Step 1: Parse form data
        st.info("📝 Parsing form data...")
        form_data = self.form_parser.parse_file(input_file_path)
        
        call_recording_url = form_data.get('call_recording').value if form_data.get('call_recording') else None
        
        status_tracker.update_stage('data_parsing', 'complete')
        status_tracker.display_status()

        # Step 2: Process audio if available
        audio_result = {'success': False}
        transcript = None
        nlp_analysis = {} # Store results from the NLPAnalyzer
        
        if call_recording_url and call_recording_url.strip():
            st.info("🎵 Processing call recording...")
            audio_result = self.audio_processor.transcribe_audio(call_recording_url, status_tracker)

            # STOP PROCESS IF TRANSCRIPTION FAILS
            if not audio_result['success']:
                st.error("🚫 PROCESS STOPPED: Transcription failed. Please check the audio URL and try again.")
                status_tracker.display_status()  # Show final failed state
                return "Process stopped due to transcription failure", "error.txt"
            
            if audio_result['success']:
                transcript = audio_result['transcript']
                
                # Step 3: Fast NLP Analysis (Motivation only)
                with st.spinner("🤖 Analyzing conversation with fast NLP..."):
                    nlp_analysis = self.nlp_analyzer.analyze_transcript(transcript)

                # --- AI ANALYSIS FOR ALL MAJOR FIELDS ---
                st.info("🧠 STARTING DEEPSEEK AI ANALYSIS...")
                
                # PROPERTY TYPE CLEANING (NEW)
                with st.spinner("🧠 Cleaning 'Property Type' with AI..."):
                    current_property_type = form_data.get('property_type').value if form_data.get('property_type') else ""
                    if current_property_type and current_property_type != "Property Type Not Specified":
                        # Use the form data for property type cleaning, not transcript
                        ai_property_type = self.rephraser.rephrase("Property Type", current_property_type)
                        # Only update if we got a valid response
                        if ai_property_type and "Transcript too short" not in ai_property_type:
                            form_data['property_type'] = FieldData(
                                value=ai_property_type,
                                source='conversation', 
                                confidence=0.95
            )
                    
                # REASON FOR SELLING
                with st.spinner("🧠 Analyzing 'Reason for Selling' with AI..."):
                    ai_reason = self.rephraser.rephrase("Reason for Selling", transcript)
                    nlp_analysis['reason'] = ai_reason
                
                # PROPERTY CONDITION  
                with st.spinner("🧠 Analyzing 'Property Condition' with AI..."):
                    ai_condition = self.rephraser.rephrase("Property Condition", transcript)
                    nlp_analysis['condition'] = ai_condition
                
                # MORTGAGE STATUS
                with st.spinner("🧠 Analyzing 'Mortgage Status' with AI..."):
                    ai_mortgage = self.rephraser.rephrase("Mortgage Status", transcript)
                    nlp_analysis['mortgage'] = ai_mortgage
                
                # OCCUPANCY STATUS
                with st.spinner("🧠 Analyzing 'Occupancy Status' with AI..."):
                    ai_occupancy = self.rephraser.rephrase("Occupancy Status", transcript)
                    nlp_analysis['tenant'] = ai_occupancy

                # In the process_single_property_lead method, add this after the other AI analyses:
                # IMPORTANT CALL HIGHLIGHTS
                with st.spinner("🧠 Identifying important call highlights..."):
                    ai_highlights = self.rephraser.rephrase("Important Highlights", transcript)
                    nlp_analysis['highlights'] = ai_highlights

                # SELLER PERSONALITY
                with st.spinner("🧠 Analyzing 'Seller Personality' with AI..."):
                    ai_personality = self.rephraser.rephrase("Seller Personality", transcript)
                    nlp_analysis['personality'] = ai_personality
                
                st.success("✅ DEEPSEEK AI ANALYSIS COMPLETE")

                status_tracker.update_stage('ai_analysis', 'complete')
                status_tracker.display_status()
                
                # Step 4: Validate and override form data with AI insights
                with st.spinner("🔄 Applying AI conversation insights to form data..."):
                    form_data = self._apply_conversation_insights(form_data, nlp_analysis)
                
            else:
                st.error(f"❌ Audio processing failed: {audio_result.get('error')}")
        else:
            st.warning("⚠️ No call recording URL found in form data. Skipping audio analysis.")
            # Initialize empty nlp_analysis to avoid errors
            nlp_analysis = {
                'reason': "No transcript available",
                'condition': "No transcript available", 
                'mortgage': "No transcript available",
                'tenant': "No transcript available",
                'motivation': "No transcript available",
                'personality': "No transcript available"
            }

        # Step 5: Merge/Derive remaining fields
        with st.spinner("🔗 Deriving dependent fields (Moving Time)..."):
            form_data = self.data_merger.merge(form_data, transcript, audio_result)

        st.info("⚖️ Starting final AI-powered lead qualification...")
        qualification_results = self.ai_qualifier.qualify(form_data)
        status_tracker.update_stage('qualification', 'complete')
        status_tracker.display_status()
        
        # Step 6: Generate enhanced report
        st.info("📊 Generating final report...")
        report_content = self.report_generator.generate_report(
            form_data, 
            transcript, 
            audio_result,
            nlp_analysis,
            qualification_results,
            input_filename
        )
        status_tracker.update_stage('report_generation', 'complete')
        status_tracker.display_status()
        
        # Save report
        output_filename = self.report_generator.save_report(report_content, input_filename)
        
        # --- MAIN TABS FOR RESULTS ---
        st.subheader("🎉 PROCESSING RESULTS")
        
        # Create main tabs for different result sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 FORM DATA", 
            "🎵 TRANSCRIPT", 
            "🤖 AI ANALYSIS", 
            "⚖️ QUALIFICATION", 
            "📄 FINAL REPORT"
        ])
        
        with tab1:
            # FORM DATA TAB
            st.header("📋 Parsed Form Data")
            form_display = []
            for field_key, field_data in form_data.items():
                if field_data.value and field_data.value not in ["", "Not mentioned"]:
                    display_name = next((names[0] for key, names in self.form_parser.field_patterns.items() if key == field_key), field_key)
                    form_display.append(f"◇{display_name}: {field_data.value}")
            
            st.text_area("Form Data", "\n".join(form_display), height=500, key="form_data_tab")
        with tab2:
            # TRANSCRIPT TAB
            st.header("🎵 Call Transcript")
            if transcript and transcript != "No transcription available":
                st.text_area("Transcript", transcript, height=500, key="transcript_tab")
            else:
                st.info("No transcript available for this lead")
        
        with tab3:
            # AI ANALYSIS TAB
            st.header("🤖 AI Conversation Analysis")
            
            ai_analysis_content = []
            
            # Add all AI analysis results
            if 'reason' in nlp_analysis:
                ai_analysis_content.append(f"◇Reason for Selling: {nlp_analysis['reason']}")
            if 'condition' in nlp_analysis:
                ai_analysis_content.append(f"◇Property Condition: {nlp_analysis['condition']}")
            if 'mortgage' in nlp_analysis:
                ai_analysis_content.append(f"◇Mortgage Status: {nlp_analysis['mortgage']}")
            if 'tenant' in nlp_analysis:
                ai_analysis_content.append(f"◇Occupancy Status: {nlp_analysis['tenant']}")
            if 'personality' in nlp_analysis:
                ai_analysis_content.append(f"◇Seller Personality: {nlp_analysis['personality']}")
            if 'motivation' in nlp_analysis:
                ai_analysis_content.append(f"◇Motivation Analysis: {nlp_analysis['motivation']}")
            if 'highlights' in nlp_analysis:  
                ai_analysis_content.append(f"◇Important Call Highlights:\n{nlp_analysis['highlights']}")  
            
            if ai_analysis_content:
                st.text_area("AI Analysis Results", "\n".join(ai_analysis_content), height=500, key="ai_analysis_tab")
            else:
                st.info("No AI analysis available (no transcript)")
        
        with tab4:
            # QUALIFICATION TAB
            st.header("⚖️ Lead Qualification")
            
            qual_content = [
                f"◇Total Score: {qualification_results['total_score']}/100",
                f"◇Verdict: {qualification_results['verdict']}",
                "",
                "BREAKDOWN:"
            ]
            
            # Add breakdown
            for category, data in qualification_results['breakdown'].items():
                qual_content.append(f"◇{category.title()}: {data['score']} pts - {data['notes']}")
            
            st.text_area("Qualification Results", "\n".join(qual_content), height=500, key="qualification_tab")
        
        with tab5:
            # FINAL REPORT TAB
            st.header("📄 Final Comprehensive Report")
            st.text_area("Complete Report", report_content, height=500, key="final_report_tab")
        
        st.success(f"✅ Enhanced report generated!")
        
        # Download button
        st.download_button(
            label="⬇️ Download Complete Report",
            data=report_content,
            file_name=output_filename,
            mime="text/plain"
        )
        
        return report_content, output_filename
    
    def _apply_conversation_insights(self, form_data: Dict[str, FieldData], 
                                     nlp_analysis: Dict[str, str]) -> Dict[str, FieldData]:
        """
        Apply validated conversation insights - AI ALWAYS WINS for major fields
        """
        
        # REASON FOR SELLING: AI always wins
        conversation_reason = nlp_analysis.get('reason', '')
        if conversation_reason and "no reason" not in conversation_reason.lower():
            cleaned_reason = self.data_validator.clean_reason_field(conversation_reason)
            form_data['reason_for_selling'] = FieldData(
                value=cleaned_reason, 
                source='conversation',
                confidence=1.0
            )
        
        # PROPERTY CONDITION: AI always wins  
        conversation_condition = nlp_analysis.get('condition', '')
        if conversation_condition and "no specific" not in conversation_condition.lower():
            form_data['condition'] = FieldData(
                value=conversation_condition,
                source='conversation',
                confidence=1.0
            )
        
        # MORTGAGE: AI always wins
        conversation_mortgage = nlp_analysis.get('mortgage', '')
        if conversation_mortgage and "no mortgage information" not in conversation_mortgage.lower():
            form_data['mortgage'] = FieldData(
                value=conversation_mortgage, 
                source='conversation',
                confidence=0.95
            )
        
        # OCCUPANCY: AI always wins
        conversation_occupancy = nlp_analysis.get('tenant', '')
        if conversation_occupancy and "no occupancy information" not in conversation_occupancy.lower():
            form_data['occupancy'] = FieldData(
                value=conversation_occupancy, 
                source='conversation',
                confidence=0.95
            )
        
        # MOTIVATION: NLP analysis wins
        conversation_motivation = nlp_analysis.get('motivation', '')
        if conversation_motivation and "no motivation" not in conversation_motivation.lower():
            form_data['motivation_details'] = FieldData(
                value=conversation_motivation,
                source='conversation',
                confidence=0.9
            )
        conversation_highlights = nlp_analysis.get('highlights', '')
        if conversation_highlights and "no critical highlights" not in conversation_highlights.lower():
            nlp_analysis['highlights'] = conversation_highlights  # Ensure it's preserved
        
        return form_data


# --- STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🤖 Real Estate Lead Automation System")
st.markdown("Paste your lead data below or upload a file")

# Check if API key is available in secrets
if 'DEEPSEEK_API_KEY' in st.secrets:
    st.success("✅ DeepSeek API key found in secrets")
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
else:
    st.error("❌ DeepSeek API key not found in secrets.")
    st.stop()

# --- Property Count Selection ---
st.subheader("🏠 Property Selection")
property_count = st.radio(
    "How many properties in this lead?",
    ["1 Property", "2+ Properties"],
    horizontal=True,
    help="Choose 1 Property for standard processing, 2+ Properties for AI-powered multi-property analysis"
)
# --- Input Method Selection ---
input_method = st.radio(
    "Choose input method:",
    ["📝 Paste Lead Data", "📁 Upload File"],
    horizontal=True
)

lead_data = None
source_name = "direct_input"

if input_method == "📝 Paste Lead Data":
    st.subheader("📝 Paste Lead Form Data")
    lead_text = st.text_area(
        "Paste your lead form data here:",
        height=300,
        placeholder="Paste your lead form data in this format:\n◇List Name:-\n◇Property Type:-\n◇Seller Name:-\n◇Phone Number:-\n◇Address:-\n◇Zillow link:-\n◇Asking Price:-\n◇Zillow Estimate:-\n◇Realtor Estimate:-\n◇Redfin Estimate:-\n◇Reason For Selling:-\n◇Motivation details:-\n◇Mortgage:-\n◇Condition:-\n◇Occupancy:-\n◇Closing time:-\n◇Moving time:-\n◇Best time to call back:-\n◇Agent Name:-\n◇Call recording:-",
        label_visibility="collapsed"
    )
    
    if lead_text.strip():
        lead_data = lead_text
        source_name = "pasted_data"

else:  # File upload method
    st.subheader("📁 Upload Lead File")
    uploaded_file = st.file_uploader("Select a `.txt` lead file", type=["txt"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        lead_data = uploaded_file.getvalue().decode('utf-8')
        source_name = uploaded_file.name

# --- Process Button ---
if lead_data:
    st.markdown("---")
    
    # Show preview
    with st.expander("📋 Data Preview", expanded=True):
        st.text(lead_data[:1000] + "..." if len(lead_data) > 1000 else lead_data)
    
    if st.button("🚀 Process Lead", type="primary"):
        status_container = st.container()
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
            tmp_file.write(lead_data)
            temp_file_path = tmp_file.name

        try:
            # Initialize and run the system
            system = RealEstateAutomationSystem()
            
            # Process the lead using your existing method
            with st.container():
                if property_count == "1 Property":
                    report_content, report_filename = system.process_single_property_lead(temp_file_path, source_name)
                else:  # "2+ Properties"
                    report_content, report_filename = system.process_multi_property_lead(temp_file_path, source_name)
        
            # Success message
            st.success("✅ Lead processing completed! Check the tabs above for detailed results.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

