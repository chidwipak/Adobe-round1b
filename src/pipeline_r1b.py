import json
import fitz  # PyMuPDF
from pathlib import Path
import logging
import time
import re
from typing import Dict, List, Optional, Any
from model_utils import ModelManager
import numpy as np
from collections import defaultdict
from datetime import datetime
import unicodedata
import langdetect

class Round1BPipeline:
    def __init__(self, models_dir: str = "/opt/models"):
        self.model_manager = ModelManager(models_dir)
        self.logger = logging.getLogger(__name__)
        try:
            self.model_manager.load_all_models()
            self.logger.info("Models loaded successfully for Round 1B")
        except Exception as e:
            self.logger.warning(f"Failed to load models: {e}. Using fallback methods.")

    def process_batch(self, input_dir: str, output_dir: str, persona_file: Optional[str] = None) -> bool:
        """
        Process batch of documents for persona-driven analysis.
        Ensures strict compliance with Adobe challenge specifications.
        """
        start_time = time.time()
        try:
            # Load persona configuration
            persona = self._load_persona(persona_file)
            self.logger.info(f"Processing documents for persona: {persona['name']}")
            
            # Validate input requirements
            input_path = Path(input_dir)
            pdf_files = list(input_path.glob("*.pdf"))
            if len(pdf_files) < 3:
                self.logger.error(f"Round 1B requires at least 3 PDF files, found {len(pdf_files)}")
                return False
            
            self.logger.info(f"Processing {len(pdf_files)} documents for persona analysis")
            
            # Extract sections from all documents
            all_sections = []
            document_metadata = []
            
            for pdf_file in pdf_files:
                self.logger.info(f"Processing {pdf_file.name}")
                doc_sections = self._extract_sections_robust(str(pdf_file), persona)
                all_sections.extend(doc_sections)
                
                # Create comprehensive document metadata
                doc_metadata = {
                    "document_id": pdf_file.stem,
                    "document_name": pdf_file.name,
                    "total_pages": self._get_page_count(str(pdf_file)),
                    "file_size": pdf_file.stat().st_size,
                    "processing_timestamp": datetime.utcnow().isoformat() + "Z"
                }
                document_metadata.append(doc_metadata)
            
            # Apply robust ranking with diversity constraints
            ranked_sections = self._rank_sections_robust(all_sections, persona)
            
            # Generate robust subsection analysis
            subsection_analysis = self._generate_subsection_analysis_robust(ranked_sections, persona)
            
            # Create output with strict schema compliance
            output = {
                "metadata": {
                    "persona": persona["name"],
                    "job_to_be_done": persona["job_to_be_done"],
                    "documents": document_metadata,
                    "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                    "total_sections_extracted": len(all_sections),
                    "total_sections_ranked": len(ranked_sections),
                    "processing_time_seconds": round(time.time() - start_time, 2)
                },
                "extracted_sections": ranked_sections[:10],  # Top 10 sections
                "subsection_analysis": subsection_analysis
            }
            
            # Validate output structure
            if not self._validate_output_schema(output):
                self.logger.error("Output validation failed")
                return False
            
            # Save output with proper formatting
            output_path = Path(output_dir) / "persona_analysis.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed {len(pdf_files)} PDFs in {processing_time:.2f}s")
            self.logger.info(f"Generated {len(ranked_sections[:10])} ranked sections and {len(subsection_analysis)} subsection analyses")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in persona-driven analysis: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def process_documents(self, pdf_files: List[str], persona: str, job_description: str, output_file: str) -> bool:
        """Process documents for API compatibility"""
        try:
            # Create persona dict from parameters
            persona_dict = {
                "name": persona,
                "job_to_be_done": job_description,
                "interests": self._extract_interests_from_job(job_description),
                "expertise_level": "intermediate"
            }
            
            # Process all sections
            all_sections = []
            document_metadata = []
            
            for pdf_file in pdf_files:
                self.logger.info(f"Processing {pdf_file}")
                doc_sections = self._extract_sections_robust(pdf_file, persona_dict)
                all_sections.extend(doc_sections)
                
                # Add document metadata
                doc_path = Path(pdf_file)
                document_metadata.append({
                    "document_id": doc_path.stem,
                    "document_name": doc_path.name,
                    "total_pages": self._get_page_count(pdf_file)
                })
            
            # Apply ML-enhanced ranking
            ranked_sections = self._rank_sections_robust(all_sections, persona_dict)
            
            # Generate sub-section analysis
            subsection_analysis = self._generate_subsection_analysis_robust(ranked_sections, persona_dict)
            
            # Create output
            output = {
                "metadata": {
                    "persona": persona,
                    "job_to_be_done": job_description,
                    "documents": document_metadata
                },
                "extracted_sections": ranked_sections[:10],
                "subsection_analysis": subsection_analysis
            }
            
            # Save output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in process_documents: {e}")
            return False
    
    def _extract_interests_from_job(self, job_description: str) -> List[str]:
        """Extract interests from job description"""
        interests = []
        job_lower = job_description.lower()
        
        # Define interest categories
        interest_categories = {
            "machine learning": ["machine learning", "ml", "ai", "artificial intelligence"],
            "deep learning": ["deep learning", "neural networks", "cnn", "rnn", "transformer"],
            "active learning": ["active learning", "semi-supervised", "few-shot"],
            "computer vision": ["computer vision", "image", "vision", "detection"],
            "nlp": ["natural language", "nlp", "text", "language"],
            "data science": ["data science", "analytics", "statistics"],
            "optimization": ["optimization", "gradient", "backpropagation"],
            "research": ["research", "experiment", "evaluation", "benchmark"]
        }
        
        for category, keywords in interest_categories.items():
            if any(keyword in job_lower for keyword in keywords):
                interests.append(category)
        
        # Add default interests if none found
        if not interests:
            interests = ["machine learning", "deep learning", "active learning"]
        
        return interests

    def _get_page_count(self, pdf_path: str) -> int:
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception:
            return 0
    
    def _load_persona(self, persona_file: Optional[str]) -> Dict[str, Any]:
        if persona_file and Path(persona_file).exists():
            try:
                with open(persona_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading persona file: {e}")
        return {
            "name": "Data Scientist",
            "job_to_be_done": "Implement active learning for a deep learning model",
            "interests": ["machine learning", "deep learning", "active learning", "neural networks"],
            "expertise_level": "intermediate"
        }
    
    def _extract_sections_robust(self, pdf_path: str, persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Robust section extraction with simple, reliable text processing.
        """
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                # Simple section extraction by paragraphs
                paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip() and len(p.strip()) > 50]
                
                for i, paragraph in enumerate(paragraphs):
                    # Skip very short paragraphs
                    if len(paragraph) < 30:
                        continue
                    
                    # Create a simple title from the first sentence
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    title = sentences[0][:100] if sentences else paragraph[:100]
                    
                    # Clean up title
                    title = re.sub(r'\s+', ' ', title).strip()
                    
                    # Use the paragraph as text (limit length)
                    text = paragraph[:500] if len(paragraph) > 500 else paragraph
                    
                    # Calculate relevance score
                    relevance_score = self._calculate_section_relevance(text, persona)
                    
                    sections.append({
                        "document": Path(pdf_path).name,
                        "page": page_num + 1,
                        "title": title,
                        "text": text,
                        "relevance_score": relevance_score
                    })
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error extracting sections from {pdf_path}: {e}")
        
        # Remove duplicates and filter quality
        seen = set()
        unique_sections = []
        for s in sections:
            key = (s["title"].strip().lower(), s["page"])
            if key not in seen and len(s["text"]) > 30:
                seen.add(key)
                unique_sections.append(s)
        
        return unique_sections

    def _calculate_section_relevance(self, text: str, persona: Dict[str, Any]) -> float:
        """
        Calculate relevance score for a section based on persona and job requirements.
        """
        try:
            # Create query from persona
            query = f"{persona['name']} {persona['job_to_be_done']} {' '.join(persona.get('interests', []))}"
            
            # Get embeddings
            query_emb = self.model_manager.get_embedding(query)
            text_emb = self.model_manager.get_embedding(text)
            
            # Calculate cosine similarity
            similarity = np.dot(query_emb, text_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(text_emb) + 1e-8)
            
            # Boost score for sections with persona keywords
            persona_keywords = persona.get('interests', []) + persona['job_to_be_done'].lower().split()
            keyword_boost = sum(1 for keyword in persona_keywords if keyword.lower() in text.lower()) * 0.1
            
            return min(1.0, float(similarity) + keyword_boost)
            
        except Exception as e:
            self.logger.warning(f"Error calculating relevance: {e}")
            return 0.0

    def _rank_sections_enhanced(self, sections: List[Dict[str, Any]], persona: Dict[str, Any], pdf_files) -> List[Dict[str, Any]]:
        """
        Enhanced section ranking with diversity constraints and better relevance scoring.
        """
        # Group sections by document
        doc_sections = defaultdict(list)
        for section in sections:
            doc_sections[section["document"]].append(section)
        
        # Ensure diversity: select top sections from each document first
        diverse_sections = []
        
        # Always include the best section from each document
        for doc, secs in doc_sections.items():
            secs_sorted = sorted(secs, key=lambda x: x.get("relevance_score", 0), reverse=True)
            if secs_sorted:
                best_section = secs_sorted[0].copy()
                diverse_sections.append(best_section)
        
        # Sort diverse sections by relevance score
        diverse_sections.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Fill remaining slots with globally best sections (avoiding duplicates)
        N = 10
        already_included = set((s["document"], s["title"]) for s in diverse_sections)
        
        remaining = [s for s in sorted(sections, key=lambda x: x.get("relevance_score", 0), reverse=True)
                     if (s["document"], s["title"]) not in already_included]
        
        while len(diverse_sections) < N and remaining:
            diverse_sections.append(remaining.pop(0))
        
        # Assign importance ranks
        for i, section in enumerate(diverse_sections):
            section["importance_rank"] = i + 1
            # Remove internal fields not needed in output
            section.pop("relevance_score", None)
            section.pop("level", None)
            section.pop("lang", None)
        
        return diverse_sections

    def _generate_subsection_analysis_enhanced(self, ranked_sections: List[Dict[str, Any]], persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate enhanced subsection analysis with meaningful explanations and refined text.
        """
        subsections = []
        
        for section in ranked_sections[:5]:  # Top 5 sections
            # Get full section text for better analysis
            full_text = self._find_full_section_text(section)
            
            # Generate refined text using summarization
            refined_text = self.model_manager.generate_summary(full_text, max_length=250)
            
            # Find the most relevant persona interest or job keyword
            best_match, best_score = self._find_best_relevance_match(full_text, persona)
            
            # Generate meaningful explanation
            explanation = self._generate_relevance_explanation(best_match, best_score, section["title"], persona)
            
            # Create subsection analysis
            subsection = {
                "document": section["document"],
                "page": section["page"],
                "parent_section": section["title"],
                "refined_text": refined_text,
                "relevance_explanation": explanation,
                "relevance_score": round(best_score, 3)
            }
            
            subsections.append(subsection)
        
        return subsections

    def _find_best_relevance_match(self, text: str, persona: Dict[str, Any]) -> tuple:
        """
        Find the most relevant persona interest or job keyword in the text.
        """
        best_match = None
        best_score = -1
        
        try:
            text_emb = self.model_manager.get_embedding(text)
            
            # Check persona interests
            for interest in persona.get("interests", []):
                interest_emb = self.model_manager.get_embedding(interest)
                sim = float((text_emb @ interest_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(interest_emb) + 1e-8))
                if sim > best_score:
                    best_score = sim
                    best_match = interest
            
            # Check job description keywords
            job_words = [word for word in persona.get("job_to_be_done", "").split() if len(word) > 3]
            for word in job_words:
                word_emb = self.model_manager.get_embedding(word)
                sim = float((text_emb @ word_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(word_emb) + 1e-8))
                if sim > best_score:
                    best_score = sim
                    best_match = word
            
            if not best_match:
                best_match = "the persona's expertise area"
                best_score = 0.0
                
        except Exception as e:
            self.logger.warning(f"Error finding relevance match: {e}")
            best_match = "the persona's expertise area"
            best_score = 0.0
        
        return best_match, best_score

    def _generate_relevance_explanation(self, best_match: str, score: float, section_title: str, persona: Dict[str, Any]) -> str:
        """
        Generate a meaningful explanation of why the section is relevant.
        """
        if score > 0.7:
            explanation = f"This section is highly relevant because it directly addresses {best_match} (similarity: {score:.2f}), which is a key requirement for the {persona['name']} role."
        elif score > 0.5:
            explanation = f"This section is relevant as it discusses {best_match} (similarity: {score:.2f}), which aligns with the {persona['name']} position requirements."
        elif score > 0.3:
            explanation = f"This section contains information related to {best_match} (similarity: {score:.2f}), which may be useful for the {persona['name']} role."
        else:
            explanation = f"This section provides general information that could be relevant to the {persona['name']} position, though the connection to specific requirements is less direct."
        
        return explanation

    def _find_full_section_text(self, section: Dict[str, Any]) -> str:
        """
        Find the full section text for analysis.
        """
        try:
            doc_path = Path("input") / section["document"]
            if not doc_path.exists():
                doc_path = Path("/app/input") / section["document"]
            
            doc = fitz.open(str(doc_path))
            page = doc[section["page"] - 1]
            text = page.get_text("text")
            doc.close()
            
            return text[:1500]  # Limit for processing efficiency
            
        except Exception as e:
            self.logger.warning(f"Error finding full section text: {e}")
            return section.get("title", "")

    def _validate_output_schema(self, output: Dict) -> bool:
        """
        Validate that the output conforms to the Adobe challenge schema.
        """
        try:
            # Check required top-level keys
            required_keys = ["metadata", "extracted_sections", "subsection_analysis"]
            if not all(key in output for key in required_keys):
                self.logger.error("Missing required top-level keys")
                return False
            
            # Validate metadata
            metadata = output["metadata"]
            required_metadata = ["persona", "job_to_be_done", "documents", "processing_timestamp"]
            if not all(key in metadata for key in required_metadata):
                self.logger.error("Missing required metadata fields")
                return False
            
            # Validate extracted sections
            sections = output["extracted_sections"]
            if not isinstance(sections, list):
                self.logger.error("extracted_sections must be a list")
                return False
            
            for i, section in enumerate(sections):
                required_section_keys = ["document", "page", "title", "importance_rank"]
                if not all(key in section for key in required_section_keys):
                    self.logger.error(f"Missing required keys in section {i}")
                    return False
            
            # Validate subsection analysis
            subsections = output["subsection_analysis"]
            if not isinstance(subsections, list):
                self.logger.error("subsection_analysis must be a list")
                return False
            
            for i, subsection in enumerate(subsections):
                required_subsection_keys = ["document", "page", "parent_section", "refined_text", "relevance_explanation"]
                if not all(key in subsection for key in required_subsection_keys):
                    self.logger.error(f"Missing required keys in subsection {i}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return False

    def _rank_sections_robust(self, sections: List[Dict[str, Any]], persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Robust section ranking with proper model inference and validation.
        """
        import numpy as np
        
        # Create comprehensive query from persona
        query = f"{persona['name']} {persona['job_to_be_done']} {' '.join(persona.get('interests', []))}"
        self.logger.info(f"Ranking sections using query: {query[:100]}...")
        
        # Get query embedding once
        try:
            query_emb = self.model_manager.get_embedding(query)
            self.logger.info(f"Query embedding shape: {query_emb.shape}, non-zero elements: {np.count_nonzero(query_emb)}")
        except Exception as e:
            self.logger.error(f"Failed to get query embedding: {e}")
            return sections
        
        # Calculate relevance scores for each section
        for i, section in enumerate(sections):
            try:
                section_text = section.get("text", "")
                if not section_text or len(section_text.strip()) < 10:
                    section["relevance_score"] = 0.0
                    continue
                
                # Get section embedding
                section_emb = self.model_manager.get_embedding(section_text)
                
                # Validate embeddings
                if np.all(section_emb == 0) or np.linalg.norm(section_emb) == 0:
                    self.logger.warning(f"Zero embedding for section {i} in {section['document']}")
                    section["relevance_score"] = 0.0
                    continue
                
                # Calculate cosine similarity
                sim = float((query_emb @ section_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(section_emb) + 1e-8))
                
                # Ensure score is in valid range
                sim = max(0.0, min(1.0, sim))
                section["relevance_score"] = sim
                
                # Log for debugging
                if i < 3:  # Log first 3 sections
                    self.logger.info(f"Section {i}: {section['document']} - Score: {sim:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Error calculating relevance for section {i}: {e}")
                section["relevance_score"] = 0.0
        # Group sections by document
        doc_sections = defaultdict(list)
        for section in sections:
            doc_sections[section["document"]].append(section)
        # For each document, select the top section(s)
        top_sections = []
        for doc, secs in doc_sections.items():
            secs_sorted = sorted(secs, key=lambda x: x.get("relevance_score", 0), reverse=True)
            if secs_sorted:
                top_sections.append(secs_sorted[0])  # Always include the best from each doc
                # Optionally, include more per doc if desired (e.g., top 2)
        # Fill the rest of the top N by global score, but avoid duplicates
        N = 10
        already_included = set((s["document"], s["title"]) for s in top_sections)
        remaining = [s for s in sorted(sections, key=lambda x: x.get("relevance_score", 0), reverse=True)
                     if (s["document"], s["title"]) not in already_included]
        while len(top_sections) < N and remaining:
            top_sections.append(remaining.pop(0))
        # Assign importance_rank
        for i, section in enumerate(top_sections):
            section["importance_rank"] = i + 1
        return top_sections

    def _generate_subsection_analysis_robust(self, ranked_sections: List[Dict[str, Any]], persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate enhanced subsection analysis with meaningful explanations and refined text.
        """
        import numpy as np
        subsections = []
        
        # Data Scientist specific skills and keywords
        ds_skills = [
            "machine learning", "deep learning", "neural networks", "statistical analysis", 
            "data analysis", "python", "pytorch", "tensorflow", "scikit-learn", "pandas", 
            "numpy", "matplotlib", "seaborn", "jupyter", "sql", "data visualization", 
            "feature engineering", "model evaluation", "cross-validation", "hyperparameter tuning",
            "active learning", "computer vision", "natural language processing", "nlp",
            "reinforcement learning", "unsupervised learning", "supervised learning",
            "data preprocessing", "data cleaning", "feature selection", "dimensionality reduction"
        ]
        
        for section in ranked_sections[:5]:
            self.logger.info(f"Processing subsection analysis for: {section['document']}")
            
            # 1. Create a clear, complete parent_section title
            parent_section = self._create_clear_section_title(section["title"], section["text"])
            
            # 2. Generate meaningful refined text with proper summarization
            refined_text = self._generate_meaningful_summary(section["text"], persona)
            
            # 3. Calculate semantic similarity with proper model inference and validation
            try:
                section_emb = self.model_manager.get_embedding(section["text"])
                self.logger.info(f"Section embedding shape: {section_emb.shape}, non-zero elements: {np.count_nonzero(section_emb)}")
                
                # Validate embedding quality
                if np.all(section_emb == 0) or np.linalg.norm(section_emb) == 0:
                    self.logger.warning(f"Zero embedding detected for section in {section['document']}")
                    section_emb = None
                else:
                    self.logger.info(f"Valid embedding generated for {section['document']}")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for {section['document']}: {e}")
                section_emb = None
            
            # 4. Find the most relevant Data Scientist skill for this section
            if section_emb is not None:
                best_match, best_score = self._find_most_relevant_ds_skill(section["text"], section_emb, ds_skills, persona)
            else:
                best_match, best_score = self._find_most_relevant_ds_skill_fallback(section["text"], ds_skills, persona)
            
            # 5. Generate specific, detailed relevance explanation
            explanation = self._generate_detailed_relevance_explanation(
                section["text"], best_match, best_score, section["document"], persona
            )
            
            # 6. Log the results for debugging
            self.logger.info(f"Best match: {best_match}, Score: {best_score:.3f}")
            self.logger.info(f"Refined text length: {len(refined_text)}")
            
            subsections.append({
                "document": section["document"],
                "page": section["page"],
                "parent_section": parent_section,
                "refined_text": refined_text,
                "relevance_explanation": explanation,
                "relevance_score": best_score
            })
        
        return subsections
    
    def _create_clear_section_title(self, title: str, text: str) -> str:
        """Create a clear, complete section title"""
        # Clean up the title
        title = title.strip()
        
        # If title is too long, truncated, or unclear, create a better one
        if len(title) > 100 or title.endswith('...') or title.endswith('.') or len(title) < 10:
            # Extract meaningful title from text
            lines = text.split('\n')
            meaningful_lines = []
            
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if len(line) > 15 and len(line) < 100:
                    # Skip lines that are just formatting or too generic
                    skip_words = ['page', 'chapter', 'section', 'abstract', 'introduction', 'brief', 'profile']
                    if not any(skip in line.lower() for skip in skip_words):
                        # Look for lines that seem like titles (capitalized, not too long)
                        if line[0].isupper() and not line.isupper():
                            meaningful_lines.append(line)
            
            if meaningful_lines:
                # Take the first meaningful line as title
                title = meaningful_lines[0]
            else:
                # Fallback: create title from first meaningful sentence
                sentences = text.split('.')
                for sentence in sentences[:3]:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and len(sentence) < 80:
                        # Clean up the sentence for title
                        title = sentence.replace('\n', ' ').strip()
                        break
                else:
                    # Last resort: create a descriptive title
                    title = f"Section from {text[:50]}..."
        
        # Final cleanup
        title = title.replace('\n', ' ').strip()
        if len(title) > 80:
            title = title[:77] + "..."
        
        return title
    
    def _generate_meaningful_summary(self, text: str, persona: Dict[str, Any]) -> str:
        """Generate meaningful summary using proper summarization"""
        if not text or len(text.strip()) < 20:
            return text
        
        try:
            # Use the model manager's summarization
            summary = self.model_manager.generate_summary(text, max_length=300)
            
            # Validate summary quality
            if len(summary) < 50 or summary == text[:len(summary)] or summary.strip() == text.strip()[:len(summary.strip())]:
                # Create a more meaningful summary using extractive approach
                summary = self._create_extractive_summary(text, persona)
            
            # Ensure summary is complete and meaningful
            if len(summary) < 30:
                summary = self._create_extractive_summary(text, persona)
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Failed to generate summary: {e}")
            return self._create_extractive_summary(text, persona)
    
    def _create_extractive_summary(self, text: str, persona: Dict[str, Any]) -> str:
        """Create extractive summary using sentence selection and keyword matching"""
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Score sentences based on relevance to persona
        persona_keywords = persona.get('interests', []) + persona.get('job_to_be_done', '').split()
        sentence_scores = []
        
        for sentence in sentences:
            if len(sentence) < 20:
                continue
                
            # Score based on keyword presence
            score = 0
            sentence_lower = sentence.lower()
            for keyword in persona_keywords:
                if keyword.lower() in sentence_lower:
                    score += 1
            
            # Bonus for longer, more informative sentences
            if len(sentence) > 50:
                score += 0.5
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 3-4 sentences
        selected_sentences = []
        for sentence, score in sentence_scores[:4]:
            if score > 0 or len(selected_sentences) < 2:  # Always include at least 2 sentences
                selected_sentences.append(sentence)
        
        if selected_sentences:
            summary = '. '.join(selected_sentences) + '.'
            return summary[:400]  # Limit length
        
        # Fallback: take first few meaningful sentences
        meaningful_sentences = [s for s in sentences if len(s) > 30][:3]
        if meaningful_sentences:
            return '. '.join(meaningful_sentences) + '.'
        
        # Last resort: return first part of text
        return text[:300] + ('...' if len(text) > 300 else '')
    
    def _find_most_relevant_ds_skill(self, text: str, section_emb: np.ndarray, ds_skills: List[str], persona: Dict[str, Any]) -> tuple:
        """Find the most relevant Data Scientist skill for this section"""
        best_match = None
        best_score = -1
        
        try:
            # Check against Data Scientist specific skills
            for skill in ds_skills:
                skill_emb = self.model_manager.get_embedding(skill)
                sim = float((section_emb @ skill_emb) / (np.linalg.norm(section_emb) * np.linalg.norm(skill_emb) + 1e-8))
                
                if sim > best_score:
                    best_score = sim
                    best_match = skill
            
            # Also check persona interests
            for interest in persona.get("interests", []):
                interest_emb = self.model_manager.get_embedding(interest)
                sim = float((section_emb @ interest_emb) / (np.linalg.norm(section_emb) * np.linalg.norm(interest_emb) + 1e-8))
                
                if sim > best_score:
                    best_score = sim
                    best_match = interest
            
            # Check job description keywords
            job_words = [word for word in persona.get("job_to_be_done", "").split() if len(word) > 3]
            for word in job_words:
                word_emb = self.model_manager.get_embedding(word)
                sim = float((section_emb @ word_emb) / (np.linalg.norm(section_emb) * np.linalg.norm(word_emb) + 1e-8))
                
                if sim > best_score:
                    best_score = sim
                    best_match = word
            
            # Ensure we have a valid score
            if best_score < 0:
                best_score = 0.0
                best_match = "data analysis"
            
            return best_match, best_score
            
        except Exception as e:
            self.logger.warning(f"Error finding relevant DS skill: {e}")
            return "data analysis", 0.0
    
    def _find_most_relevant_ds_skill_fallback(self, text: str, ds_skills: List[str], persona: Dict[str, Any]) -> tuple:
        """Fallback method to find relevant skills when embeddings fail"""
        text_lower = text.lower()
        best_match = "data analysis"
        best_score = 0.0
        
        # Check for skill mentions in text
        for skill in ds_skills:
            if skill.lower() in text_lower:
                # Count occurrences and calculate a simple score
                count = text_lower.count(skill.lower())
                score = min(0.8, count * 0.2)  # Cap at 0.8
                
                if score > best_score:
                    best_score = score
                    best_match = skill
        
        # Check persona interests
        for interest in persona.get("interests", []):
            if interest.lower() in text_lower:
                count = text_lower.count(interest.lower())
                score = min(0.9, count * 0.25)  # Slightly higher weight for interests
                
                if score > best_score:
                    best_score = score
                    best_match = interest
        
        # Check job description keywords
        job_words = [word for word in persona.get("job_to_be_done", "").split() if len(word) > 3]
        for word in job_words:
            if word.lower() in text_lower:
                count = text_lower.count(word.lower())
                score = min(0.85, count * 0.3)  # Higher weight for job-related words
                
                if score > best_score:
                    best_score = score
                    best_match = word
        
        return best_match, best_score
    
    def _generate_detailed_relevance_explanation(self, text: str, best_match: str, score: float, document: str, persona: Dict[str, Any]) -> str:
        """Generate detailed, specific relevance explanation"""
        # Extract key information from the text
        text_lower = text.lower()
        
        # Identify specific skills mentioned in the text
        mentioned_skills = []
        ds_skills = ["python", "pytorch", "tensorflow", "pandas", "numpy", "machine learning", "deep learning", "neural networks", "statistical analysis", "data visualization", "active learning", "cnn", "convolutional", "gradient", "optimization"]
        
        for skill in ds_skills:
            if skill in text_lower:
                mentioned_skills.append(skill)
        
        # Extract key concepts from the text
        key_concepts = []
        if "active learning" in text_lower:
            key_concepts.append("active learning strategies")
        if "neural network" in text_lower or "deep learning" in text_lower:
            key_concepts.append("deep learning models")
        if "gradient" in text_lower:
            key_concepts.append("gradient-based optimization")
        if "batch" in text_lower:
            key_concepts.append("batch processing")
        if "uncertainty" in text_lower:
            key_concepts.append("uncertainty quantification")
        
        # Create detailed explanation based on content and relevance
        if score > 0.6:
            if mentioned_skills:
                skills_str = ", ".join(mentioned_skills[:3])
                if key_concepts:
                    concepts_str = ", ".join(key_concepts[:2])
                    explanation = f"This section from {document} is highly relevant for Data Scientists implementing {persona.get('job_to_be_done', 'active learning')} because it directly addresses {best_match} (relevance: {score:.2f}) and specifically covers {skills_str}. The content discusses {concepts_str}, providing practical methodologies that Data Scientists can apply to reduce data annotation costs and improve model performance through intelligent sample selection."
                else:
                    explanation = f"This section from {document} is highly relevant for Data Scientists because it directly addresses {best_match} (relevance: {score:.2f}) and specifically mentions {skills_str}. This content provides practical knowledge and techniques that Data Scientists can apply in their work, particularly in areas like model development, data analysis, and statistical modeling."
            else:
                explanation = f"This section from {document} is highly relevant for Data Scientists because it directly addresses {best_match} (relevance: {score:.2f}). The content provides valuable insights and methodologies that align with core Data Scientist responsibilities including machine learning implementation, statistical analysis, and data-driven decision making."
        
        elif score > 0.4:
            if key_concepts:
                concepts_str = ", ".join(key_concepts[:2])
                explanation = f"This section from {document} is relevant for Data Scientists as it discusses {best_match} (relevance: {score:.2f}) and covers {concepts_str}. The content offers useful information that can enhance a Data Scientist's understanding of {best_match} and its applications in real-world scenarios, particularly for {persona.get('job_to_be_done', 'active learning')}."
            else:
                explanation = f"This section from {document} is relevant for Data Scientists as it discusses {best_match} (relevance: {score:.2f}), which is an important skill area in data science. The content offers useful information that can enhance a Data Scientist's understanding of {best_match} and its applications in real-world scenarios."
        
        elif score > 0.2:
            explanation = f"This section from {document} contains information related to {best_match} (relevance: {score:.2f}) that may be useful for Data Scientists. While the connection is moderate, the content provides general knowledge that could be valuable for understanding broader context in data science projects and {persona.get('job_to_be_done', 'active learning')}."
        
        else:
            explanation = f"This section from {document} provides general information that could be useful for Data Scientists in understanding various aspects of their field. While the direct connection to specific Data Scientist skills is limited, the content offers valuable context and knowledge that supports professional development in {persona.get('job_to_be_done', 'active learning')}."
        
        return explanation 

    def _rank_sections_challenge(self, sections: List[Dict[str, Any]], persona: Dict[str, Any], pdf_files) -> List[Dict[str, Any]]:
        # Use embedding similarity and cross-encoder for robust ranking
        query = f"{persona['name']} {persona['job_to_be_done']} {' '.join(persona.get('interests', []))}"
        for section in sections:
            try:
                if self.model_manager.models.get("cross_encoder") is not None:
                    pairs = [(query, section.get("text", ""))]
                    score = self.model_manager.models["cross_encoder"].predict(pairs)[0]
                    section["relevance_score"] = float(score)
                else:
                    emb1 = self.model_manager.get_embedding(query)
                    emb2 = self.model_manager.get_embedding(section.get("text", ""))
                    sim = float((emb1 @ emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
                    section["relevance_score"] = sim
            except Exception:
                section["relevance_score"] = 0.0
        # Group by document, always include at least one from each
        doc_sections = defaultdict(list)
        for section in sections:
            doc_sections[section["document"]].append(section)
        top_sections = []
        for doc, secs in doc_sections.items():
            secs_sorted = sorted(secs, key=lambda x: x.get("relevance_score", 0), reverse=True)
            if secs_sorted:
                # Use the best section from each doc
                s = secs_sorted[0]
                top_sections.append({
                    "document": s["document"],
                    "page": s["page"],
                    "title": s["title"],
                    "text": s["text"],
                    "importance_rank": 0  # will be set later
                })
        # Fill the rest of the top N by global score, avoid duplicates
        N = 10
        already_included = set((s["document"], s["title"]) for s in top_sections)
        remaining = [s for s in sorted(sections, key=lambda x: x.get("relevance_score", 0), reverse=True)
                     if (s["document"], s["title"]) not in already_included]
        while len(top_sections) < N and remaining:
            s = remaining.pop(0)
            top_sections.append({
                "document": s["document"],
                "page": s["page"],
                "title": s["title"],
                "text": s["text"],
                "importance_rank": 0
            })
        # Assign importance_rank
        for i, section in enumerate(top_sections):
            section["importance_rank"] = i + 1
        return top_sections
        
    def _generate_subsection_analysis_challenge(self, ranked_sections: List[Dict[str, Any]], persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        subsections = []
        for section in ranked_sections[:5]:
            # Find the full section text for this document/title/page
            full_text = self._find_full_section_text(section)
            refined_text = self.model_manager.generate_summary(full_text, max_length=200)
            # Use embedding similarity to find the most relevant interest or job keyword
            best_match = None
            best_score = -1
            query_emb = self.model_manager.get_embedding(full_text)
            for interest in persona.get("interests", []):
                interest_emb = self.model_manager.get_embedding(interest)
                sim = float((query_emb @ interest_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(interest_emb) + 1e-8))
                if sim > best_score:
                    best_score = sim
                    best_match = interest
            if not best_match:
                for word in persona.get("job_to_be_done", "").split():
                    if len(word) > 3:
                        word_emb = self.model_manager.get_embedding(word)
                        sim = float((query_emb @ word_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(word_emb) + 1e-8))
                        if sim > best_score:
                            best_score = sim
                            best_match = word
            if not best_match:
                best_match = "the persona's interests/job"
            explanation = f"This section is relevant because it discusses {best_match} (similarity: {best_score:.2f}) in the context of the persona's needs."
            subsections.append({
                "document": section["document"],
                "page": section["page"],
                "parent_section": section["title"],
                "refined_text": refined_text,
                "relevance_explanation": explanation
            })
        return subsections
    
    def _find_full_section_text(self, section: Dict[str, Any]) -> str:
        # Find the full section text for a given document/title/page in all_sections
        # This is a simple implementation; in a real system, you might want to cache or pass all_sections
        # For now, just re-extract from the PDF
        try:
            doc_path = Path("input") / section["document"]
            if not doc_path.exists():
                doc_path = Path("/app/input") / section["document"]
            doc = fitz.open(str(doc_path))
            page = doc[section["page"] - 1]
            text = page.get_text("text")
            doc.close()
            return text[:1000]  # Truncate for summary
        except Exception:
            return section.get("title", "") 