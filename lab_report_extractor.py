import re
import warnings
from typing import Dict, Any, List, Optional, Tuple
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from transformers import pipeline
    import torch
    HAS_BERT = True
except ImportError:
    HAS_BERT = False

class BERTExtractor:
    def __init__(self):
        self.ner_pipeline = None
        self.medical_ner_pipeline = None
        self.device = "cpu"
        
        if HAS_BERT:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        
        if HAS_BERT:
            try:
                # Use clinical BERT for medical NER
                self.medical_ner_pipeline = pipeline(
                    "ner", 
                    model="emilyalsentzer/Bio_ClinicalBERT", 
                    aggregation_strategy="simple",
                    device=0 if self.device == "cuda" else -1
                )
                print(f"✓ Clinical BERT loaded successfully on {self.device}")
            except Exception as e:
                print(f"✗ Failed to load clinical BERT: {e}")
            
            try:
                # Use general BERT for general NER
                self.ner_pipeline = pipeline(
                    "ner", 
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if self.device == "cuda" else -1
                )
                print(f"✓ General BERT loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load general BERT: {e}")

    # ==================== PATIENT INFORMATION EXTRACTION ====================
    
    def extract_patient_info_bert(self, text: str) -> Dict[str, Any]:
        """Extract patient information using BERT NER with enhanced pattern recognition"""
        patient_info = {
            'name': 'UNKNOWN',
            'age': 'UNKNOWN', 
            'sex': 'UNKNOWN',
            'dob': 'UNKNOWN',
            'date': 'UNKNOWN',
            'visit_id': 'UNKNOWN'
        }
        
        # First try regex-based extraction (more reliable for structured data)
        regex_result = self.extract_patient_info_regex(text)
        
        # If we have a good pipeline, enhance with BERT
        if self.ner_pipeline or self.medical_ner_pipeline:
            try:
                # Focus on first 20 lines for patient info (increased from 15)
                lines = text.split('\n')[:20]
                header_text = ' '.join(lines)
                
                # Process in chunks to avoid token limits
                chunks = [header_text[i:i+400] for i in range(0, len(header_text), 300)]
                
                for chunk in chunks[:3]:  # Process first 3 chunks
                    # Try both pipelines
                    entities = []
                    if self.medical_ner_pipeline:
                        try:
                            medical_entities = self.medical_ner_pipeline(chunk)
                            entities.extend(medical_entities)
                        except:
                            pass
                    
                    if self.ner_pipeline:
                        try:
                            general_entities = self.ner_pipeline(chunk)
                            entities.extend(general_entities)
                        except:
                            pass
                    
                    for entity in entities:
                        entity_text = entity['word'].strip()
                        entity_type = entity['entity_group']
                        confidence = entity['score']
                        
                        # Extract names (PERSON entities or high-confidence MISC)
                        if entity_type == 'PER' or (entity_type == 'MISC' and confidence > 0.5):
                            if self._is_valid_name(entity_text):
                                patient_info['name'] = entity_text.upper()
                        
                        # Also try to extract names from context
                        if patient_info['name'] == 'UNKNOWN':
                            if self._is_likely_name_in_context(entity_text, chunk):
                                patient_info['name'] = entity_text.upper()
                        
                        # Extract dates (DATE entities)
                        elif entity_type == 'DATE':
                            if self._is_valid_date(entity_text):
                                if 'dob' in chunk.lower() or 'birth' in chunk.lower():
                                    patient_info['dob'] = entity_text
                                else:
                                    patient_info['date'] = entity_text
                        
                        # Extract ages (look for numbers with age context)
                        elif self._is_age_entity(entity_text, chunk):
                            age_match = re.search(r'(\d+)', entity_text)
                            if age_match and 1 <= int(age_match.group(1)) <= 120:
                                patient_info['age'] = int(age_match.group(1))
                        
                        # Extract sex/gender
                        elif self._is_sex_entity(entity_text):
                            sex = entity_text.lower()
                            patient_info['sex'] = 'M' if sex in ['male', 'm'] else 'F'
                        
                        # Extract visit/patient IDs
                        elif self._is_id_entity(entity_text, chunk):
                            patient_info['visit_id'] = entity_text
                
                # Merge with regex results (regex takes precedence for structured data)
                for key, value in regex_result.items():
                    if value != 'UNKNOWN' and value is not None:
                        patient_info[key] = value
                
                return patient_info
                
            except Exception as e:
                print(f"BERT extraction failed: {e}")
                return regex_result
        
        return regex_result
    
    def extract_patient_info_regex(self, text: str) -> Dict[str, Any]:
        """Enhanced regex-based patient info extraction for any lab report format"""
        patient_info = {
            'name': 'UNKNOWN',
            'age': 'UNKNOWN', 
            'sex': 'UNKNOWN',
            'dob': 'UNKNOWN',
            'date': 'UNKNOWN',
            'visit_id': 'UNKNOWN'
        }
        
        # First try header section (first 30 lines)
        lines = text.split('\n')
        header_lines = lines[:30]
        header_text = ' '.join(header_lines)
        
        # Extract each field from header
        patient_info['name'] = self._extract_name(header_text, header_lines)
        patient_info['age'] = self._extract_age(header_text)
        patient_info['sex'] = self._extract_sex(header_text)
        patient_info['dob'] = self._extract_dob(header_text)
        patient_info['date'] = self._extract_date(header_text)
        patient_info['visit_id'] = self._extract_visit_id(header_text)
        
        # If we didn't find patient info in header, search the entire document
        if (patient_info['name'] == 'UNKNOWN' or 
            patient_info['age'] == 'UNKNOWN' or 
            patient_info['sex'] == 'UNKNOWN'):
            
            full_text = ' '.join(lines)
            
            # Try to find patient info anywhere in the document
            if patient_info['name'] == 'UNKNOWN':
                patient_info['name'] = self._extract_name(full_text, lines)
            
            if patient_info['age'] == 'UNKNOWN':
                patient_info['age'] = self._extract_age(full_text)
            
            if patient_info['sex'] == 'UNKNOWN':
                patient_info['sex'] = self._extract_sex(full_text)
            
            if patient_info['date'] == 'UNKNOWN':
                patient_info['date'] = self._extract_date(full_text)
            
            if patient_info['visit_id'] == 'UNKNOWN':
                patient_info['visit_id'] = self._extract_visit_id(full_text)
        
        return patient_info
    
    def _extract_name(self, header_text: str, lines: List[str]) -> str:
        """Extract patient name using multiple strategies"""
        
        # Strategy 1: Look for name patterns that are clearly patient names (not test names)
        # Pattern: Name field followed by actual name
        name_field_pattern = r'Name\s*[:\-]?\s*(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)'
        name_field_match = re.search(name_field_pattern, header_text, re.IGNORECASE)
        if name_field_match:
            name = self._clean_name(name_field_match.group(1))
            if self._is_valid_name(name):
                return name
        
        # Strategy 2: Handle format like "MRS. SATHYAVATHY ." (case insensitive)
        mrs_pattern = r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*\.'
        mrs_match = re.search(mrs_pattern, header_text, re.IGNORECASE)
        if mrs_match:
            name = self._clean_name(mrs_match.group(1))
            if self._is_valid_name(name):
                return name
        
        # Strategy 3: Handle format like "MRS. SATHYAVATHY ." (with period at end, case insensitive)
        mrs_pattern2 = r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*\.\s*$'
        mrs_match2 = re.search(mrs_pattern2, header_text, re.MULTILINE | re.IGNORECASE)
        if mrs_match2:
            name = self._clean_name(mrs_match2.group(1))
            if self._is_valid_name(name):
                return name
        
        # Strategy 4: Handle specific PDF format like "Mrs. SWARNA LATA:"
        pdf_name_pattern = r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\s*:'
        pdf_match = re.search(pdf_name_pattern, header_text)
        if pdf_match:
            name = self._clean_name(pdf_match.group(1))
            if self._is_valid_name(name):
                return name
        
        # Strategy 5: Explicit name patterns with labels
        label_patterns = [
            r'(?:patient\s*name|name|patient)\s*[:\-]?\s*(?:mr\.?|mrs\.?|ms\.?|dr\.?)?\s*([A-Z][A-Za-z\s\.]{4,50})(?=\s*(?:Age|Sex|DOB|Date|ID|Visit|MRD|Lab|Test|\d{2}|$))',
            r'(?:mr\.?|mrs\.?|ms\.?|dr\.?)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)(?=\s*(?:Age|Sex|DOB|Date|ID|Visit|MRD|Lab|Test|\d{2}|$))',
        ]
        
        for pattern in label_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = self._clean_name(match.group(1))
                if self._is_valid_name(name):
                    return name
        
        # Strategy 2: All caps names (common in Indian labs)
        caps_pattern = r'\b([A-Z][A-Z\s]{10,50})\b(?=\s*(?:Age|Sex|DOB|Date|ID|Visit|MRD|\d{2}|$))'
        caps_match = re.search(caps_pattern, header_text)
        if caps_match:
            name = self._clean_name(caps_match.group(1))
            if self._is_valid_name(name) and len(name.split()) >= 2:
                return name.title()
        
        # Strategy 3: Line-by-line analysis for standalone names
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if len(line) < 5 or len(line) > 60:
                continue
            
            # Look for lines that look like names (but not test names)
            if re.match(r'^(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*[A-Z][A-Za-z\s\.]{5,50}$', line):
                # Skip if it looks like a test name
                if any(test_word in line.upper() for test_word in ['TEST', 'FUNCTION', 'TOTAL', 'DIRECT', 'INDIRECT', 'PROTEIN', 'ALBUMIN', 'GLOBULIN', 'RATIO', 'SGOT', 'SGPT', 'GGT', 'PHOSPHATASE', 'BILIRUBIN']):
                    continue
                
                name = self._clean_name(line)
                if self._is_valid_name(name) and len(name.split()) >= 2:
                    return name
        
        # Strategy 4: Name before age/sex pattern
        context_pattern = r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\s+(?:Age|DOB|Sex|Gender|\d{1,3}\s*(?:Y|yrs|years))'
        context_match = re.search(context_pattern, header_text, re.IGNORECASE)
        if context_match:
            name = self._clean_name(context_match.group(1))
            if self._is_valid_name(name):
                return name
        
        # Strategy 5: Handle specific PDF format like "Mrs. SWARNA LATA:"
        pdf_name_pattern = r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\s*:'
        pdf_match = re.search(pdf_name_pattern, header_text)
        if pdf_match:
            name = self._clean_name(pdf_match.group(1))
            if self._is_valid_name(name):
                return name
        
        # Strategy 6: Handle format like "MRS. SATHYAVATHY ." (case insensitive)
        mrs_pattern = r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*\.'
        mrs_match = re.search(mrs_pattern, header_text, re.IGNORECASE)
        if mrs_match:
            name = self._clean_name(mrs_match.group(1))
            if self._is_valid_name(name):
                return name
        
        # Strategy 6b: Handle format like "MRS. SATHYAVATHY ." (with period at end, case insensitive)
        mrs_pattern2 = r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*\.\s*$'
        mrs_match2 = re.search(mrs_pattern2, header_text, re.MULTILINE | re.IGNORECASE)
        if mrs_match2:
            name = self._clean_name(mrs_match2.group(1))
            if self._is_valid_name(name):
                return name
        
        # Strategy 7: Look for name patterns that are clearly patient names (not test names)
        # Pattern: Name field followed by actual name
        name_field_pattern = r'Name\s*[:\-]?\s*(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)'
        name_field_match = re.search(name_field_pattern, header_text, re.IGNORECASE)
        if name_field_match:
            name = self._clean_name(name_field_match.group(1))
            if self._is_valid_name(name):
                return name
        
        return 'UNKNOWN'
    
    def _extract_age(self, header_text: str) -> Any:
        """Extract patient age"""
        age_patterns = [
            r'age\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs?|y|Y)?',
            r'(\d{1,3})\s*(?:years?|yrs?|Y)(?:\s+old)?',
            r'(\d{1,3})\s*(?:Y|y)\s*(?:/|,|\s)',
            r'age\s*[:\-]?\s*(\d{1,3})',
            r'(\d{1,3})\s*Years',  # Handle "41 Years" format
            r'(\d{1,3})Y\s*/\s*(?:Male|Female)',  # Handle "70Y / Female" format
            r'Age/Gender\s*[:\-]?\s*(\d{1,3})Y',  # Handle "Age/Gender: 70Y" format
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 0 <= age <= 120:
                    return age
        
        return 'UNKNOWN'
    
    def _extract_sex(self, header_text: str) -> str:
        """Extract patient sex/gender"""
        sex_patterns = [
            r'(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b',
            r'\b(male|female)\b(?=\s*(?:Age|DOB|Date|ID|Visit|Lab|Test|\d|$))',
            r'(?:Age.*?)\s+(male|female|m|f)\b',
            r'/\s*(male|female|m|f)\b',
            r'\b(Female|Male)\b',  # Handle "Female" format
            r'(Female)\d+',  # Handle "Female41" format
            r'(Male)\d+',    # Handle "Male41" format
            r'\b(F|M)\b(?=\s*\d+)',  # Handle "F 41" or "M 41" format
            r'\d+Y\s*/\s*(Male|Female)',  # Handle "70Y / Female" format
        ]
        
        for pattern in sex_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                try:
                    sex = match.group(1).lower()
                    return 'M' if sex in ['male', 'm'] else 'F'
                except IndexError:
                    # Handle patterns that might not have a group
                    continue
        
        return 'UNKNOWN'
    
    def _extract_dob(self, header_text: str) -> str:
        """Extract date of birth"""
        dob_patterns = [
            r'(?:dob|date\s*of\s*birth)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(?:dob|date\s*of\s*birth)\s*[:\-]?\s*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
            r'(?:born|birth)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return 'UNKNOWN'
    
    def _extract_date(self, header_text: str) -> str:
        """Extract report/collection date"""
        date_patterns = [
            r'(?:date|collected|report\s*date|sample\s*date)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(?:date|collected|report\s*date|sample\s*date)\s*[:\-]?\s*(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})',
            r'(?:on|dated)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: find any date-like pattern
        date_match = re.search(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', header_text)
        if date_match:
            return date_match.group(1)
        
        return 'UNKNOWN'
    
    def _extract_visit_id(self, header_text: str) -> str:
        """Extract patient/visit ID"""
        id_patterns = [
            r'(?:patient\s*id|visit\s*id|id|mrd|uhid|op\s*no|ip\s*no|reg\s*no|ref\s*no)\s*[:\-]?\s*([A-Za-z0-9\-\/]{4,20})',
            r'(?:id|no)\s*[:\-]?\s*([A-Za-z0-9]{6,20})',
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return 'UNKNOWN'
    
    def _clean_name(self, name: str) -> str:
        """Clean and format name"""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Remove titles
        name = re.sub(r'^(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+', '', name, flags=re.IGNORECASE)
        
        # Remove special characters except spaces and dots
        name = re.sub(r'[^\w\s\.]', '', name)
        
        # Title case
        return name.title()
    
    def _is_valid_name(self, text: str) -> bool:
        if len(text) < 5 or len(text) > 50:
            return False
        
        # Check for invalid words (column headers and common non-name terms)
        # Use word boundaries to avoid false positives
        invalid_words = [
            r'\bREPORT\b', r'\bSTATUS\b', r'\bMALE\b', r'\bFEMALE\b', r'\bYEARS\b', 
            r'\bAGE\b', r'\bLAB\b', r'\bCOLLECTED\b', r'\bPATIENT\b', r'\bNAME\b', 
            r'\bOBSERVED\b', r'\bVALUE\b', r'\bUNIT\b', r'\bBIOLOGICAL\b', r'\bREFERENCE\b',
            r'\bINTERVAL\b', r'\bSPECIMEN\b', r'\bINVESTIGATION\b', r'\bMETHOD\b', 
            r'\bSERUM\b', r'\bPLASMA\b', r'\bBLOOD\b', r'\bURINE\b', r'\bSTOOL\b', 
            r'\bSPUTUM\b', r'\bTEST\b', r'\bRESULT\b', r'\bRESULTS\b', r'\bPRINT\b', 
            r'\bCOLLECTION\b', r'\bRECEIVED\b', r'\bHOSPITAL\b', r'\bUHID\b', 
            r'\bREFERRAL\b', r'\bGENDER\b', r'\bID\b', r'\bNO\b', r'\bNUMBER\b'
        ]
        if any(re.search(inv, text.upper()) for inv in invalid_words):
            return False
        
        # Check if it contains at least 1 word (some names might be single words)
        words = text.split()
        if len(words) < 1:
            return False
        
        # Check if it's mostly alphabetic
        if not re.match(r'^[A-Za-z\s\.]+$', text):
            return False
        
        # Additional check: should not be all caps (likely a header)
        if text.isupper() and len(text) > 10:
            return False
        
        return True
    
    def _is_valid_date(self, text: str) -> bool:
        """Check if text is a valid date"""
        date_patterns = [
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
            r'\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}'
        ]
        return any(re.search(pattern, text) for pattern in date_patterns)
    
    def _is_age_entity(self, text: str, context: str) -> bool:
        """Check if entity is likely an age"""
        age_indicators = ['age', 'years', 'yrs', 'y/o', 'old']
        return (re.search(r'\d+', text) and 
                any(indicator in context.lower() for indicator in age_indicators))
    
    def _is_sex_entity(self, text: str) -> bool:
        """Check if entity is likely a sex/gender"""
        sex_values = ['male', 'female', 'm', 'f', 'man', 'woman']
        return text.lower() in sex_values
    
    def _is_id_entity(self, text: str, context: str) -> bool:
        """Check if entity is likely a patient/visit ID"""
        id_indicators = ['id', 'patient', 'visit', 'op', 'ip', 'ref']
        return (len(text) > 3 and 
                any(indicator in context.lower() for indicator in id_indicators))
    
    def _is_likely_name_in_context(self, text: str, context: str) -> bool:
        """Check if text is likely a name based on context"""
        name_indicators = ['patient', 'name', 'mr.', 'mrs.', 'ms.']
        return (self._is_valid_name(text) and 
                any(indicator in context.lower() for indicator in name_indicators))

    # ==================== TEST EXTRACTION ====================
    
    def extract_tests_bert(self, text: str) -> List[Dict[str, Any]]:
        """Extract test results using Clinical BERT with enhanced regex fallback"""
        
        # Identify test section
        test_section = self._extract_test_section(text)
        
        all_tests = []
        
        # Primary: Clinical BERT extraction
        if self.medical_ner_pipeline:
            bert_tests = self._extract_tests_with_clinical_bert(test_section)
            all_tests.extend(bert_tests)
            print(f"✓ Clinical BERT extracted {len(bert_tests)} tests")
        
        # Secondary: Enhanced regex extraction (use full text to ensure we don't miss anything)
        regex_tests = self.extract_tests_regex(text)
        print(f"✓ Regex extracted {len(regex_tests)} tests")
        
        # Merge results (prefer Clinical BERT, supplement with regex)
        merged_tests = self._merge_test_results(all_tests, regex_tests)
        
        print(f"✓ Total unique tests extracted: {len(merged_tests)}")
        return merged_tests
    
    def extract_tests_regex(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced regex-based test extraction for any lab report format"""
        tests = []
        lines = text.split('\n')
        
        # First, try multi-line pattern matching for PDF format
        tests.extend(self._extract_multiline_tests(lines))
        
        # Enhanced patterns for different lab report formats
        patterns = [
            # Standard format: Test Name Value Unit
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s+([0-9]+\.?[0-9]*)\s+(mg/dl|g/dl|iu/l|u/l|/ul|%|mmol/l|ng/ml|pg/ml|mcg/ml|iu/ml|cells/ul|fl|pg|g%)',
            
            # Colon format: Test Name: Value Unit
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s*[:\-]\s*([0-9]+\.?[0-9]*)\s*([a-zA-Z\/\%\s\.\-\d]{1,15})',
            
            # Tabular format: Test Name    Value    Unit
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s{2,}([0-9]+\.?[0-9]*)\s{2,}([a-zA-Z\/\%\s\.\-\d]{1,15})',
            
            # Parentheses format: Test Name (Value Unit)
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s*KATEX_INLINE_OPEN([0-9]+\.?[0-9]*)\s*([a-zA-Z\/\%\s\.\-\d]{1,15})KATEX_INLINE_CLOSE',
            
            # Indian lab format: Test Name - Value Unit
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s*-\s*([0-9]+\.?[0-9]*)\s*([a-zA-Z\/\%\s\.\-\d]{1,15})',
            
            # Simple format: Test Name Value (without explicit unit)
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s+([0-9]+\.?[0-9]*)(?=\s|$)',
            
            # Abbreviation format: HB 12.5 g/dl
            r'([A-Z]{2,6})\s+([0-9]+\.?[0-9]*)\s+([a-zA-Z\/\%\s\.\-\d]{1,15})',
            
            # PDF format: Value Test Name Units Reference
            r'([0-9]+\.?[0-9]*)\s+([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s+([a-zA-Z\/\%\s\.\-\d]{1,15})',
            
            # PDF format with parentheses: Value Test Name (Units) Reference
            r'([0-9]+\.?[0-9]*)\s+([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s*KATEX_INLINE_OPEN([a-zA-Z\/\%\s\.\-\d]{1,15})KATEX_INLINE_CLOSE',
            
            # Universal patterns for more lab report formats
            # Pattern: Test Name = Value Unit
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s*=\s*([0-9]+\.?[0-9]*)\s*([a-zA-Z\/\%\s\.\-\d]{1,15})',
            
            # Pattern: Test Name Value Unit (with more flexible spacing)
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})\s+([0-9]+\.?[0-9]*)\s+([a-zA-Z\/\%\s\.\-\d]{1,15})',
            
            # Pattern: Value Test Name (reverse order)
            r'([0-9]+\.?[0-9]*)\s+([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{3,50})(?=\s|$)',
            
            # Pattern: Test Name Value (no unit, more flexible)
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{2,50})\s+([0-9]+\.?[0-9]*)(?=\s|$|\n)',
        ]
        
        for line in lines:
            if len(line.strip()) < 5:
                continue
            
            # Skip lines that are clearly not test results
            skip_keywords = ['patient', 'name', 'age', 'sex', 'date', 'report', 'hospital', 'doctor', 'address']
            if any(keyword in line.lower() for keyword in skip_keywords):
                continue
                
            for pattern_idx, pattern in enumerate(patterns):
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Handle different pattern formats
                    if pattern_idx >= 7 and pattern_idx <= 9:  # PDF format patterns (value first)
                        value = match.group(1).strip()
                        name = match.group(2).strip().title()
                        unit = match.group(3).strip() if len(match.groups()) >= 3 else ""
                    elif pattern_idx == 12:  # Reverse order pattern (value first, no unit)
                        value = match.group(1).strip()
                        name = match.group(2).strip().title()
                        unit = ""
                    else:  # Standard patterns (name first)
                        name = match.group(1).strip().title()
                        value = match.group(2).strip()
                        unit = match.group(3).strip() if len(match.groups()) >= 3 else ""
                    
                    # Clean up the name
                    name = re.sub(r'\s+', ' ', name)
                    name = re.sub(r'[^\w\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE\.]', '', name)
                    
                    # Validate the test name
                    if (len(name) >= 2 and 
                        not name.isdigit() and 
                        not any(skip in name.lower() for skip in skip_keywords) and
                        self._is_medical_test(name)):
                        
                        # Clean up the unit
                        if unit:
                            unit = re.sub(r'\s+', ' ', unit)
                            unit = re.sub(r'[^\w\s\/\%\.\-\d]', '', unit)
                        
                        tests.append({
                            'name': name,
                            'value': value,
                            'unit': unit,
                            'confidence': 0.8,
                            'matched_tokens': [name, value, unit] if unit else [name, value]
                        })
        
        # Remove duplicates and sort by confidence
        unique_tests = []
        seen = set()
        for test in sorted(tests, key=lambda x: x['confidence'], reverse=True):
            key = test['name'].lower()
            if key not in seen:
                unique_tests.append(test)
                seen.add(key)
        
        return unique_tests[:30]
    
    def _extract_multiline_tests(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract tests from multi-line format common in PDFs"""
        tests = []
        
        for i in range(len(lines) - 1):
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            
            # Pattern: "0.63 Creatinine" followed by "(...) 0.51 - 0.95 mg/dL"
            value_name_match = re.match(r'^([0-9]+\.?[0-9]*)\s+([A-Za-z][A-Za-z\s\-\.]{3,50})$', current_line)
            
            if value_name_match:
                value = value_name_match.group(1)
                name = value_name_match.group(2).strip().title()
                
                # Look for unit and reference range in next line
                unit = ""
                ref_range = ""
                
                if next_line:
                    # Extract unit and reference range from next line
                    unit_match = re.search(r'([a-zA-Z\/\%\s\.\-\d]{1,15})\s*$', next_line)
                    if unit_match:
                        unit = unit_match.group(1).strip()
                    
                    # Extract reference range
                    ref_match = re.search(r'([0-9]+\.?[0-9]*\s*-\s*[0-9]+\.?[0-9]*)', next_line)
                    if ref_match:
                        ref_range = ref_match.group(1).strip()
                
                # Clean up the name
                name = re.sub(r'\s+', ' ', name)
                name = re.sub(r'[^\w\s\-\.]', '', name)
                
                # Validate the test name
                if (len(name) >= 2 and 
                    not name.isdigit() and 
                    self._is_medical_test(name)):
                    
                    tests.append({
                        'name': name,
                        'value': value,
                        'unit': unit,
                        'reference_range': ref_range,
                        'confidence': 0.9,
                        'matched_tokens': [name, value, unit] if unit else [name, value]
                    })
        
        return tests
    
    def _extract_test_section(self, text: str) -> str:
        """Intelligently extract the test results section"""
        lines = text.split('\n')
        
        # Keywords that indicate start of test section
        start_keywords = [
            'test', 'investigation', 'result', 'parameter', 'examination',
            'biochemistry', 'hematology', 'serology', 'pathology',
            'report', 'findings', 'lab', 'value', 'unit', 'reference'
        ]
        
        # Keywords that indicate end of test section
        end_keywords = [
            'interpretation', 'comment', 'note', 'remark', 'signature',
            'doctor', 'pathologist', 'consultant', 'end of report',
            'disclaimer', 'reference', 'methodology'
        ]
        
        start_idx = 0
        end_idx = len(lines)
        
        # Find start of test section (skip patient info header)
        for i in range(min(30, len(lines))):
            line_lower = lines[i].lower()
            if any(keyword in line_lower for keyword in start_keywords):
                # Make sure it's not just in the header
                if i > 5 or any(char.isdigit() for char in lines[i]):
                    start_idx = max(0, i - 2)  # Include 2 lines before for context
                    break
        
        # Find end of test section
        for i in range(start_idx + 10, len(lines)):
            line_lower = lines[i].lower()
            if any(keyword in line_lower for keyword in end_keywords):
                end_idx = i
                break
        
        test_section = '\n'.join(lines[start_idx:end_idx])
        return test_section
    
    def _extract_tests_with_clinical_bert(self, text: str) -> List[Dict[str, Any]]:
        """Extract tests using Clinical BERT NER"""
        tests = []
        
        try:
            # Split into manageable chunks (Clinical BERT has token limits)
            chunks = self._split_into_chunks(text, max_length=450, overlap=50)
            
            for chunk_idx, chunk in enumerate(chunks[:15]):  # Process up to 15 chunks
                try:
                    # Get entities from Clinical BERT
                    entities = self.medical_ner_pipeline(chunk)
                    
                    # Process entities and find test patterns
                    for i, entity in enumerate(entities):
                        entity_text = entity['word'].strip()
                        entity_type = entity['entity_group']
                        confidence = entity['score']
                        
                        # Look for medical test indicators
                        if confidence > 0.25 and len(entity_text) > 2:
                            # Check if this could be a test name
                            if self._is_potential_test_name(entity_text, chunk):
                                # Try to find associated value and unit
                                test_info = self._extract_test_components(entity_text, chunk)
                                
                                if test_info:
                                    tests.append({
                                        'name': test_info['name'],
                                        'value': test_info['value'],
                                        'unit': test_info['unit'],
                                        'reference_range': test_info.get('reference_range', ''),
                                        'confidence': confidence,
                                        'source': 'clinical_bert',
                                        'matched_tokens': test_info.get('matched_tokens', [])
                                    })
                    
                    # Also try context-based extraction within chunk
                    context_tests = self._extract_tests_from_context(chunk, entities)
                    tests.extend(context_tests)
                    
                except Exception as e:
                    print(f"✗ Error processing chunk {chunk_idx}: {e}")
                    continue
            
        except Exception as e:
            print(f"✗ Clinical BERT extraction failed: {e}")
        
        return tests
    
    def _split_into_chunks(self, text: str, max_length: int = 450, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better context"""
        chunks = []
        lines = text.split('\n')
        
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line)
            
            if current_length + line_length > max_length and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_lines = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_lines + [line]
                current_length = sum(len(l) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_length += line_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _is_potential_test_name(self, text: str, context: str) -> bool:
        """Check if text could be a medical test name"""
        text_lower = text.lower().strip()
        
        # Length check
        if len(text_lower) < 2 or len(text_lower) > 60:
            return False
        
        # Medical test keywords
        medical_indicators = [
            # Blood tests
            'hemoglobin', 'hb', 'hgb', 'hematocrit', 'hct', 'pcv',
            'wbc', 'white blood', 'rbc', 'red blood', 'platelets', 'platelet', 'plt',
            'mcv', 'mch', 'mchc', 'rdw', 'mpv', 'pdw',
            
            # Biochemistry
            'glucose', 'sugar', 'cholesterol', 'hdl', 'ldl', 'vldl', 'triglyceride',
            'creatinine', 'urea', 'bun', 'uric acid', 'bilirubin',
            'sgot', 'sgpt', 'ast', 'alt', 'alkaline phosphatase', 'alp',
            'ggt', 'gamma', 'ldh', 'cpk', 'ck', 'troponin', 'bnp',
            
            # Electrolytes
            'sodium', 'potassium', 'chloride', 'bicarbonate', 'calcium',
            'phosphorus', 'phosphate', 'magnesium', 'iron',
            
            # Proteins
            'protein', 'albumin', 'globulin', 'ratio',
            
            # Inflammatory
            'esr', 'crp', 'c-reactive', 'procalcitonin', 'ferritin',
            
            # Hormones
            'tsh', 'thyroid', 't3', 't4', 'cortisol', 'insulin', 'hba1c',
            'testosterone', 'estrogen', 'prolactin', 'lh', 'fsh',
            
            # Kidney
                        'egfr', 'microalbumin', 'creatinine clearance',
            
            # Liver
            'pt', 'inr', 'aptt', 'prothrombin',
            
            # Cardiac
            'ck-mb', 'myoglobin', 'nt-probnp', 'nt pro bnp',
            
            # Tumor markers
            'psa', 'cea', 'ca 125', 'ca 19-9', 'afp', 'ca-125', 'ca-19-9',
            
            # Vitamins
            'vitamin', 'b12', 'folate', 'folic acid',
            
            # Common abbreviations
            'tlc', 'dlc', 'abs', 'count', 'level', 'serum', 'plasma', 'blood'
        ]
        
        # Direct match
        if any(indicator in text_lower for indicator in medical_indicators):
            return True
        
        # Pattern-based detection
        if re.match(r'^[a-z]{2,5}$', text_lower):  # Short abbreviations
            return True
        
        if re.search(r'\b(count|level|ratio|index|test)\b', text_lower):
            return True
        
        # Context check: is there a number nearby?
        context_window = context[max(0, context.find(text) - 50):context.find(text) + 100]
        if re.search(r'\d+\.?\d*', context_window):
            return True
        
        return False
    
    def _is_medical_test(self, text: str) -> bool:
        """Check if text is likely a medical test name - enhanced for any lab report"""
        medical_keywords = [
            # Blood tests
            'hemoglobin', 'hb', 'hgb', 'hematocrit', 'hct', 'pcv', 'packed cell volume',
            'wbc', 'white blood cell', 'rbc', 'red blood cell', 'platelets', 'plt',
            'mcv', 'mch', 'mchc', 'rdw', 'mpv', 'pdw',
            
            # Biochemistry
            'glucose', 'sugar', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 'tg',
            'creatinine', 'urea', 'bun', 'uric acid', 'bilirubin', 'total bilirubin',
            'direct bilirubin', 'indirect bilirubin', 'ast', 'alt', 'sgot', 'sgpt',
            'alkaline phosphatase', 'alp', 'ggt', 'gamma gt', 'lactate dehydrogenase', 'ldh',
            'cpk', 'ck', 'creatine kinase', 'troponin', 'bnp', 'pro bnp',
            
            # Electrolytes
            'sodium', 'na', 'potassium', 'k', 'chloride', 'cl', 'bicarbonate', 'hco3',
            'calcium', 'ca', 'phosphorus', 'phosphate', 'po4', 'magnesium', 'mg',
            
            # Proteins
            'total protein', 'albumin', 'globulin', 'a/g ratio', 'ag ratio',
            
            # Inflammatory markers
            'esr', 'erythrocyte sedimentation rate', 'crp', 'c reactive protein',
            'procalcitonin', 'ferritin', 'iron', 'tibc', 'transferrin',
            
            # Hormones
            'tsh', 'thyroid stimulating hormone', 't3', 't4', 'free t3', 'free t4',
            'cortisol', 'insulin', 'hba1c', 'glycated hemoglobin',
            
            # Kidney function
            'egfr', 'estimated glomerular filtration rate', 'microalbumin',
            'protein creatinine ratio', 'pcr',
            
            # Liver function
            'pt', 'prothrombin time', 'inr', 'aptt', 'activated partial thromboplastin time',
            
            # Cardiac markers
            'ck mb', 'ck-mb', 'myoglobin', 'nt pro bnp',
            
            # Tumor markers
            'psa', 'prostate specific antigen', 'cea', 'carcinoembryonic antigen',
            'ca 125', 'ca 19-9', 'afp', 'alpha fetoprotein',
            
            # Vitamins
            'vitamin d', 'vitamin b12', 'folate', 'folic acid',
            
            # Common abbreviations
            'tlc', 'total leukocyte count', 'dc', 'differential count',
            'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils',
            'serum', 'plasma', 'blood', 'urine', 'stool', 'sputum',
            
            # General terms
            'total', 'count', 'level', 'concentration', 'activity', 'ratio',
            
            # Additional medical terms for universal compatibility
            'gfr', 'estimated', 'category', 'nitrogen', 'bun', 'creatinine',
            'acid', 'phosphatase', 'dehydrogenase', 'kinase', 'phosphatase',
            'protein', 'albumin', 'globulin', 'bilirubin', 'cholesterol',
            'glucose', 'sugar', 'insulin', 'thyroid', 'hormone', 'vitamin',
            'mineral', 'electrolyte', 'enzyme', 'marker', 'indicator',
            'panel', 'profile', 'screen', 'assay', 'test', 'examination'
        ]
        
        text_lower = text.lower().strip()
        
        # Direct keyword match
        if any(keyword in text_lower for keyword in medical_keywords):
            return True
        
        # Pattern-based detection
        medical_patterns = [
            r'^[a-z]{2,6}$',  # Short abbreviations like HB, WBC, etc.
            r'.*test.*',      # Contains "test"
            r'.*level.*',     # Contains "level"
            r'.*count.*',     # Contains "count"
            r'.*ratio.*',     # Contains "ratio"
            r'.*index.*',     # Contains "index"
            r'.*panel.*',     # Contains "panel"
            r'.*profile.*',   # Contains "profile"
            r'.*screen.*',    # Contains "screen"
            r'.*assay.*',     # Contains "assay"
            r'.*examination.*', # Contains "examination"
        ]
        
        for pattern in medical_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Additional check: if it contains numbers and looks like a medical term
        if re.search(r'\d+', text) and len(text) > 2 and len(text) < 50:
            # Check if it's not just a number or date
            if not re.match(r'^\d+[\.\-\/]*\d*$', text):
                return True
        
        return False
    
    def _extract_test_components(self, test_name: str, context: str) -> Optional[Dict[str, Any]]:
        """Extract test value, unit, and reference range from context"""
        
        # Find position of test name in context
        test_pos = context.lower().find(test_name.lower())
        if test_pos == -1:
            return None
        
        # Extract relevant context (line containing test name and next line)
        context_start = max(0, test_pos - 50)
        context_end = min(len(context), test_pos + len(test_name) + 150)
        relevant_context = context[context_start:context_end]
        
        # Patterns to match test results
        patterns = [
            # Pattern 1: Test Name: Value Unit (Reference Range)
            rf'{re.escape(test_name)}\s*[:\-]?\s*(\d+\.?\d*)\s*([a-zA-Z\/\%\^\s\.\-\d³²µ]+?)(?:\s*KATEX_INLINE_OPEN([^KATEX_INLINE_CLOSE]+)KATEX_INLINE_CLOSE)?(?=\s|$|\n)',
            
            # Pattern 2: Test Name Value Unit Reference
            rf'{re.escape(test_name)}\s+(\d+\.?\d*)\s+([a-zA-Z\/\%\^\s\.\-\d³²µ]+?)\s*(\d+\.?\d*\s*-\s*\d+\.?\d*)?',
            
            # Pattern 3: Tabular format with whitespace
            rf'{re.escape(test_name)}\s{{2,}}(\d+\.?\d*)\s{{2,}}([a-zA-Z\/\%\^\s\.\-\d³²µ]+?)(?:\s{{2,}}([^\n]+?))?(?=\n|$)',
            
            # Pattern 4: Value immediately after name
            rf'{re.escape(test_name)}\s*[:\-]?\s*(\d+\.?\d*)\s*([a-zA-Z\/\%\^\s\.\-\d³²µ]*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, relevant_context, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                unit = match.group(2).strip() if len(match.groups()) >= 2 else ''
                ref_range = match.group(3).strip() if len(match.groups()) >= 3 and match.group(3) else ''
                
                # Validate value
                try:
                    float_val = float(value)
                    if not (0.0001 <= float_val <= 1000000):
                        continue
                except ValueError:
                    continue
                
                # Clean unit
                unit = self._clean_unit(unit)
                
                # Clean reference range
                ref_range = self._clean_reference_range(ref_range)
                
                return {
                    'name': test_name.strip().title(),
                    'value': value,
                    'unit': unit,
                    'reference_range': ref_range,
                    'matched_tokens': [test_name, value, unit]
                }
        
        return None
    
    def _find_test_value_unit(self, test_name: str, context: str) -> tuple:
        """Find value and unit associated with a test name"""
        # Look for patterns like "Test Name: 123.45 mg/dl" or "Test Name 123.45 mg/dl"
        patterns = [
            rf'{re.escape(test_name)}\s*[:\-]?\s*(\d+\.?\d*)\s*([a-zA-Z\/\%\s\.\-\d]+?)(?=\s*\n|\s*[A-Z]|\s*$)',
            rf'{re.escape(test_name)}\s+(\d+\.?\d*)\s+([a-zA-Z\/\%\s\.\-\d]+?)(?=\s*\n|\s*[A-Z]|\s*$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                unit = match.group(2).strip()
                
                # Clean up unit
                unit = re.sub(r'\s+', ' ', unit)
                unit = re.sub(r'[^\w\s\/\%\.\-\d]', '', unit)
                
                # Validate value
                try:
                    float_val = float(value)
                    if 0.001 <= float_val <= 10000:
                        return value, unit
                except ValueError:
                    continue
        
        return None, None
    
    def _extract_tests_from_context(self, chunk: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Extract tests by analyzing context and patterns"""
        tests = []
        
        # Look for lines that match test result patterns
        lines = chunk.split('\n')
        
        for line in lines:
            if len(line.strip()) < 5:
                continue
            
            # Pattern: TestName Value Unit
            pattern = r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.]{2,40})\s+(\d+\.?\d*)\s+([a-zA-Z\/\%\^\s\.\-\d³²µ]+)'
            matches = re.finditer(pattern, line)
            
            for match in matches:
                name = match.group(1).strip()
                value = match.group(2).strip()
                unit = match.group(3).strip()
                
                # Validate
                if self._is_potential_test_name(name, line):
                    try:
                        float(value)
                        tests.append({
                            'name': name.title(),
                            'value': value,
                            'unit': self._clean_unit(unit),
                            'reference_range': '',
                            'confidence': 0.7,
                            'source': 'clinical_bert_context',
                            'matched_tokens': [name, value, unit]
                        })
                    except ValueError:
                        continue
        
        return tests
    
    def _extract_tests_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced regex-based test extraction"""
        tests = []
        lines = text.split('\n')
        
        # Comprehensive patterns for various lab report formats
        patterns = [
            # Pattern 1: Name Value Unit (Reference)
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.\/]{2,50})\s+(\d+\.?\d*)\s+([a-zA-Z\/\%\^\s\.\-\d³²µ]+?)\s*(?:KATEX_INLINE_OPEN([^KATEX_INLINE_CLOSE]+)KATEX_INLINE_CLOSE)?(?=\s*$|\s*\n)',
            
            # Pattern 2: Name: Value Unit
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.\/]{2,50})\s*[:\-]\s*(\d+\.?\d*)\s*([a-zA-Z\/\%\^\s\.\-\d³²µ]*)',
            
            # Pattern 3: Tabular with multiple spaces
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.\/]{2,50})\s{2,}(\d+\.?\d*)\s{1,}([a-zA-Z\/\%\^\s\.\-\d³²µ]+)',
            
            # Pattern 4: Parentheses format
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.\/]{2,50})\s*KATEX_INLINE_OPEN(\d+\.?\d*)\s*([a-zA-Z\/\%\^\s\.\-\d³²µ]+)KATEX_INLINE_CLOSE',
            
            # Pattern 5: Dash separator
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.\/]{2,50})\s*-\s*(\d+\.?\d*)\s*([a-zA-Z\/\%\^\s\.\-\d³²µ]*)',
            
            # Pattern 6: Abbreviation format (e.g., "HB 12.5 g/dL")
            r'\b([A-Z]{2,6})\b\s+(\d+\.?\d*)\s+([a-zA-Z\/\%\^\s\.\-\d³²µ]+)',
            
            # Pattern 7: With "is" or "=" (e.g., "Glucose is 95 mg/dL")
            r'([A-Za-z][A-Za-z\s\-KATEX_INLINE_OPENKATEX_INLINE_CLOSE,\.\/]{2,50})\s+(?:is|=)\s+(\d+\.?\d*)\s*([a-zA-Z\/\%\^\s\.\-\d³²µ]*)',
        ]
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            if len(line) < 5:
                continue
            
            # Skip header lines
            skip_keywords = [
                'patient', 'name', 'age', 'sex', 'date', 'hospital', 
                'doctor', 'address', 'reference', 'normal', 'abnormal',
                'specimen', 'collected', 'received', 'reported'
            ]
            
            if any(keyword in line.lower()[:30] for keyword in skip_keywords):
                continue
            
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                
                for match in matches:
                    name = match.group(1).strip()
                    value = match.group(2).strip()
                    unit = match.group(3).strip() if len(match.groups()) >= 3 else ''
                    ref_range = match.group(4).strip() if len(match.groups()) >= 4 and match.group(4) else ''
                    
                    # Clean and validate name
                    name = self._clean_test_name(name)
                    
                    if not self._is_valid_test_name(name):
                        continue
                    
                    # Validate value
                    try:
                        float_val = float(value)
                        if not (0.0001 <= float_val <= 1000000):
                            continue
                    except ValueError:
                        continue
                    
                    # Clean unit
                    unit = self._clean_unit(unit)
                    
                    # Clean reference range
                    ref_range = self._clean_reference_range(ref_range)
                    
                    tests.append({
                        'name': name,
                        'value': value,
                        'unit': unit,
                        'reference_range': ref_range,
                        'confidence': 0.85,
                        'source': 'regex',
                        'matched_tokens': [name, value, unit]
                    })
        
        return tests
    
    def _clean_test_name(self, name: str) -> str:
        """Clean and standardize test name"""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Remove leading/trailing special characters
        name = re.sub(r'^[^\w]+|[^\w]+$', '', name)
        
        # Remove common prefixes
        name = re.sub(r'^(serum|plasma|blood|urine|total|free)\s+', '', name, flags=re.IGNORECASE)
        
        # Title case
        return name.title()
    
    def _is_valid_test_name(self, name: str) -> bool:
        """Validate test name"""
        if not name or len(name) < 2 or len(name) > 60:
            return False
        
        # Must contain at least one letter
        if not re.search(r'[A-Za-z]', name):
            return False
        
        # Should not be just numbers
        if name.replace('.', '').replace('-', '').isdigit():
            return False
        
        # Invalid keywords
        invalid = [
            'page', 'report', 'continued', 'end', 'signature', 'doctor',
            'pathologist', 'laboratory', 'phone', 'email', 'address',
            'normal', 'abnormal', 'high', 'low', 'reference', 'range',
            'method', 'specimen', 'sample', 'collected', 'received'
        ]
        
        name_lower = name.lower()
        if any(inv in name_lower for inv in invalid):
            return False
        
        return True
    
    def _clean_unit(self, unit: str) -> str:
        """Clean and standardize unit"""
        if not unit:
            return ''
        
        # Remove extra whitespace
        unit = re.sub(r'\s+', ' ', unit.strip())
        
        # Remove trailing non-alphanumeric characters
        unit = re.sub(r'[^\w\s\/\%\^\.\-³²µ]+$', '', unit)
        
        # Limit length
        if len(unit) > 25:
            unit = unit[:25]
        
        # Common unit standardizations
        unit_map = {
            'gm/dl': 'g/dL',
            'gm%': 'g%',
            'mg%': 'mg/dL',
            'cells/cumm': 'cells/µL',
            'per cumm': '/µL',
            '/cmm': '/µL',
            'cu mm': '/µL',
            'cumm': '/µL',
            'micro l': 'µL',
            'ul': 'µL',
            'iu/l': 'IU/L',
            'u/l': 'U/L',
            'mill': 'million',
            'thou': 'thousand',
            'pg': 'pg',
            'fl': 'fL',
        }
        
        unit_lower = unit.lower()
        for old, new in unit_map.items():
            if old in unit_lower:
                unit = unit_lower.replace(old, new)
                break
        
        return unit.strip()
    
    def _clean_reference_range(self, ref_range: str) -> str:
        """Clean and standardize reference range"""
        if not ref_range:
            return ''
        
        # Remove extra whitespace
        ref_range = re.sub(r'\s+', ' ', ref_range.strip())
        
        # Remove common prefixes
        ref_range = re.sub(r'^(ref|reference|normal|range)[:\s]*', '', ref_range, flags=re.IGNORECASE)
        
        # Limit length
        if len(ref_range) > 50:
            ref_range = ref_range[:50]
        
        return ref_range.strip()
    
    def _merge_test_results(self, bert_tests: List[Dict], regex_tests: List[Dict]) -> List[Dict[str, Any]]:
        """Merge BERT and regex results, removing duplicates and prioritizing quality"""
        
        # Create a dictionary to store unique tests
        merged = {}
        
        # Add BERT tests (higher priority)
        for test in bert_tests:
            key = self._normalize_test_key(test['name'])
            
            if key not in merged or test['confidence'] > merged[key].get('confidence', 0):
                merged[key] = test
        
        # Add regex tests (fill gaps)
        for test in regex_tests:
            key = self._normalize_test_key(test['name'])
            
            # Add if not present or if regex has better data
            if key not in merged:
                merged[key] = test
            elif not merged[key].get('unit') and test.get('unit'):
                # Prefer test with unit information
                merged[key] = test
            elif not merged[key].get('reference_range') and test.get('reference_range'):
                # Add reference range if missing
                merged[key]['reference_range'] = test['reference_range']
        
        # Convert back to list and sort by confidence
        result = list(merged.values())
        result.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Limit to top 50 tests
        return result[:50]
    
    def _normalize_test_key(self, name: str) -> str:
        """Normalize test name for comparison"""
        # Convert to lowercase
        key = name.lower().strip()
        
        # Remove common variations
        key = re.sub(r'\s+', '', key)  # Remove all spaces
        key = re.sub(r'[^\w]', '', key)  # Remove special characters
        
        # Remove common prefixes
        prefixes = ['serum', 'plasma', 'blood', 'urine', 'total', 'free']
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
        
        return key
    
    # ==================== MAIN EXTRACTION METHOD ====================
    
    def extract_with_text(self, text: str) -> Dict[str, Any]:
        """Main extraction method combining all strategies"""
        
        if len(text.strip()) < 10:
            return {
                "patient": {
                    "name": "UNKNOWN", 
                    "age": "UNKNOWN", 
                    "sex": "UNKNOWN",
                    "dob": "UNKNOWN", 
                    "visit_id": "UNKNOWN", 
                    "date": "UNKNOWN"
                },
                "tests": []
            }
        
        print(f"\n{'='*60}")
        print(f"Starting extraction on text of length: {len(text)}")
        print(f"{'='*60}\n")
        
        # Extract patient information (regex-based)
        print("📋 Extracting patient information...")
        patient_info = self.extract_patient_info_regex(text)
        print(f"✓ Patient info extracted: {patient_info['name']}, Age: {patient_info['age']}, Sex: {patient_info['sex']}")
        
        # Extract test results (Clinical BERT + regex)
        print("\n🔬 Extracting test results...")
        tests = self.extract_tests_bert(text)
        
        # Post-process tests
        tests = self._post_process_tests(tests)
        
        print(f"\n{'='*60}")
        print(f"✓ Extraction complete: {len(tests)} tests extracted")
        print(f"{'='*60}\n")
        
        return {
            "patient": patient_info, 
            "tests": tests
        }
    
    def _post_process_tests(self, tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process and validate extracted tests"""
        
        processed = []
        
        for test in tests:
            # Ensure all required fields exist
            processed_test = {
                'name': test.get('name', 'UNKNOWN'),
                'value': test.get('value', 'UNKNOWN'),
                'unit': test.get('unit', ''),
                'reference_range': test.get('reference_range', ''),
                'confidence': round(test.get('confidence', 0.5), 3),
                'source': test.get('source', 'unknown')
            }
            
            # Skip if invalid
            if processed_test['name'] == 'UNKNOWN' or processed_test['value'] == 'UNKNOWN':
                continue
            
            # Validate value is numeric
            try:
                float(processed_test['value'])
            except ValueError:
                continue
            
            processed.append(processed_test)
        
        return processed
    
    # ==================== UTILITY METHODS ====================
    
    def get_extraction_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the extraction"""
        tests = result.get('tests', [])
        
        stats = {
            'total_tests': len(tests),
            'tests_with_units': sum(1 for t in tests if t.get('unit')),
            'tests_with_reference': sum(1 for t in tests if t.get('reference_range')),
            'avg_confidence': round(sum(t.get('confidence', 0) for t in tests) / len(tests), 3) if tests else 0,
            'sources': {}
        }
        
        # Count by source
        for test in tests:
            source = test.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        return stats
    
    def format_output(self, result: Dict[str, Any]) -> str:
        """Format extraction result as readable string"""
        output = []
        
        # Patient information
        output.append("=" * 60)
        output.append("PATIENT INFORMATION")
        output.append("=" * 60)
        
        patient = result['patient']
        output.append(f"Name:     {patient['name']}")
        output.append(f"Age:      {patient['age']}")
        output.append(f"Sex:      {patient['sex']}")
        output.append(f"DOB:      {patient['dob']}")
        output.append(f"Date:     {patient['date']}")
        output.append(f"Visit ID: {patient['visit_id']}")
        
        # Test results
        output.append("\n" + "=" * 60)
        output.append("TEST RESULTS")
        output.append("=" * 60)
        
        tests = result['tests']
        if tests:
            # Create formatted table
            output.append(f"\n{'Test Name':<35} {'Value':<12} {'Unit':<15} {'Confidence':<10}")
            output.append("-" * 72)
            
            for test in tests:
                name = test['name'][:34]
                value = test['value'][:11]
                unit = test['unit'][:14]
                conf = f"{test['confidence']:.2f}"
                
                output.append(f"{name:<35} {value:<12} {unit:<15} {conf:<10}")
            
            # Statistics
            stats = self.get_extraction_stats(result)
            output.append("\n" + "=" * 60)
            output.append("EXTRACTION STATISTICS")
            output.append("=" * 60)
            output.append(f"Total Tests:              {stats['total_tests']}")
            output.append(f"Tests with Units:         {stats['tests_with_units']}")
            output.append(f"Tests with Reference:     {stats['tests_with_reference']}")
            output.append(f"Average Confidence:       {stats['avg_confidence']:.3f}")
            output.append(f"Sources: {stats['sources']}")
        else:
            output.append("\nNo tests found.")
        
        output.append("\n" + "=" * 60)
        
        return "\n".join(output)


# ==================== GLOBAL INSTANCE ====================

extractor = BERTExtractor()


# ==================== PUBLIC API ====================

def extract_with_text(text: str) -> Dict[str, Any]:
    """
    Extract patient information and test results from lab report text.
    
    Args:
        text: Raw text from lab report
        
    Returns:
        Dictionary containing patient info and test results
    """
    return extractor.extract_with_text(text)


def extract_and_format(text: str) -> str:
    """
    Extract and return formatted output.
    
    Args:
        text: Raw text from lab report
        
    Returns:
        Formatted string with patient info and test results
    """
    result = extractor.extract_with_text(text)
    return extractor.format_output(result)


def get_extraction_stats(text: str) -> Dict[str, Any]:
    """
    Get extraction statistics.
    
    Args:
        text: Raw text from lab report
        
    Returns:
        Dictionary with extraction statistics
    """
    result = extractor.extract_with_text(text)
    return extractor.get_extraction_stats(result)