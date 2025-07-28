import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Any, Optional
import logging
import numpy as np
import re
from collections import defaultdict

class PDFExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Patterns for common section numbering
        self.section_patterns = [
            r"^\d+\.(?:\d+\.)*\s+",  # 1. or 1.1. or 1.1.1.
            r"^[A-Z]\.(?:\d+\.)*\s+",  # A. or A.1.
            r"^[IVXLCivxlc]+\.\s+",  # Roman numerals: I. II. III. IV.
            r"^(?:Section|SECTION|Chapter|CHAPTER)\s+\d+\.?\s*",  # Section 1 or Chapter 2
            r"^(?:Appendix|APPENDIX)\s+[A-Z]\.?\s*"  # Appendix A
        ]
        
        # Common heading terms that indicate a section heading
        self.heading_terms = [
            "introduction", "abstract", "background", "method", "methodology", "approach", 
            "experiment", "evaluation", "result", "discussion", "conclusion", "reference", 
            "appendix", "acknowledgment", "related work", "future work", "implementation"
        ]
        
        # Font size thresholds based on document analysis
        self.title_size_ratio = 1.5  # Title is 1.5x larger than average
        self.h1_size_ratio = 1.3     # H1 is 1.3x larger than average
        self.h2_size_ratio = 1.15    # H2 is 1.15x larger than average
        self.h3_size_ratio = 1.05    # H3 is 1.05x larger than average
    
    def extract_text_and_layout(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text, layout, and font information from PDF"""
        try:
            doc = None
            try:
                doc = fitz.open(pdf_path)
                
                # Extract document title from metadata or first page text
                title = self._extract_document_title(doc)
                
                # First pass: analyze document statistics
                doc_stats = self._analyze_document_statistics(doc)
                
                # Second pass: extract text with layout information
                pages_data = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract text with detailed layout information
                    blocks = page.get_text("dict")
                    text_elements = []
                    
                    # Process blocks in reading order
                    for block in blocks["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                # Combine spans in the same line for better heading detection
                                line_text = ""
                                line_fonts = []
                                line_sizes = []
                                line_flags = []
                                bbox = None
                                
                                for span in line["spans"]:
                                    if span["text"].strip():
                                        if not line_text:  # First span in line
                                            bbox = list(span["bbox"])
                                        else:
                                            # Expand bounding box
                                            bbox[0] = min(bbox[0], span["bbox"][0])
                                            bbox[1] = min(bbox[1], span["bbox"][1])
                                            bbox[2] = max(bbox[2], span["bbox"][2])
                                            bbox[3] = max(bbox[3], span["bbox"][3])
                                            
                                        line_text += span["text"]
                                        line_fonts.append(span["font"])
                                        line_sizes.append(span["size"])
                                        line_flags.append(span["flags"])
                                
                                if line_text.strip():
                                    # Use most common font properties for the line
                                    most_common_size = max(set(line_sizes), key=line_sizes.count) if line_sizes else 0
                                    most_common_flags = max(set(line_flags), key=line_flags.count) if line_flags else 0
                                    
                                    text_elements.append({
                                        "text": line_text.strip(),
                                        "bbox": bbox,
                                        "font": line_fonts[0] if line_fonts else "",  # Use first font for simplicity
                                        "size": most_common_size,
                                        "flags": most_common_flags,
                                        "page": page_num + 1,
                                        "x0": bbox[0] if bbox else 0,
                                        "y0": bbox[1] if bbox else 0,
                                        "x1": bbox[2] if bbox else 0,
                                        "y1": bbox[3] if bbox else 0
                                    })
                    
                    # Sort text elements by vertical position
                    text_elements.sort(key=lambda x: x["y0"])
                    
                    pages_data.append({
                        "page_num": page_num + 1,
                        "text_elements": text_elements,
                        "page_width": page.rect.width,
                        "page_height": page.rect.height
                    })
                
                # Post-process to merge fragmented headings
                pages_data = self._merge_fragmented_headings(pages_data)
                
                return {
                    "pages": pages_data,
                    "total_pages": len(doc),
                    "title": title,
                    "statistics": doc_stats
                }
            finally:
                if doc:
                    doc.close()
                    
        except Exception as e:
            self.logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return {"pages": [], "total_pages": 0, "title": "Unknown Document", "statistics": {}}
    
    def _extract_document_title(self, doc) -> str:
        """Extract document title using multiple strategies"""
        # Strategy 1: Use document metadata
        metadata = doc.metadata
        if metadata and metadata.get("title") and len(metadata.get("title").strip()) > 3:
            return metadata.get("title").strip()
        
        # Strategy 2: Extract from first page text using sort=True for better results
        try:
            # Get text from first page with proper sorting for academic papers
            page = doc[0]
            text = page.get_text("text", sort=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Academic papers often have title in all caps or as one of the first lines
            for i, line in enumerate(lines[:10]):
                # Skip very short lines or lines that are likely not titles
                if len(line) < 10:
                    continue
                    
                # Skip lines that are likely not titles
                if any(x in line.lower() for x in ["abstract", "introduction", "http", "www.", "©", "copyright"]):
                    continue
                
                # Titles in academic papers are often in ALL CAPS or Title Case
                if line.isupper() or (i <= 3 and len(line) > 15):
                    # For academic papers, the title is often the first substantial line in all caps
                    return line
            
            # If we can't find a clear title, try to extract from blocks
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            # Look for large font text near the top of the page
                            if len(text) > 10 and span["size"] > 14 and span["bbox"][1] < page.rect.height * 0.3:
                                return text
            
            # If still no title found, use the first substantial line
            for line in lines[:5]:
                if len(line) > 15:
                    return line
                    
        except Exception as e:
            self.logger.warning(f"Error extracting title from text: {e}")
        
        # Strategy 3: For academic papers, try to find DEEP BATCH ACTIVE LEARNING pattern
        try:
            page = doc[0]
            text = page.get_text("text", sort=True)
            
            # Look for common academic paper title patterns
            match = re.search(r"([A-Z][A-Z\s]{10,})", text)
            if match:
                return match.group(1).strip()
        except Exception:
            pass
        
        # Fallback
        return "Unknown Document"
    
    def _analyze_document_statistics(self, doc) -> Dict[str, Any]:
        """Analyze document to extract font statistics and other metadata"""
        font_sizes = []
        font_counts = defaultdict(int)
        bold_counts = 0
        italic_counts = 0
        total_elements = 0
        
        for page_num in range(min(10, len(doc))):  # Analyze first 10 pages at most
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(span["size"])
                                font_counts[span["font"]] += 1
                                if span["flags"] & 16:  # Bold flag
                                    bold_counts += 1
                                if span["flags"] & 2:   # Italic flag
                                    italic_counts += 1
                                total_elements += 1
        
        # Calculate statistics
        stats = {
            "avg_font_size": np.mean(font_sizes) if font_sizes else 12,
            "median_font_size": np.median(font_sizes) if font_sizes else 12,
            "std_font_size": np.std(font_sizes) if font_sizes else 0,
            "most_common_font": max(font_counts.items(), key=lambda x: x[1])[0] if font_counts else None,
            "bold_ratio": bold_counts / max(total_elements, 1),
            "italic_ratio": italic_counts / max(total_elements, 1),
            "font_size_distribution": np.percentile(font_sizes, [25, 50, 75, 90, 95]).tolist() if len(font_sizes) > 5 else None
        }
        
        return stats
    
    def _merge_fragmented_headings(self, pages_data: List[Dict]) -> List[Dict]:
        """Merge text elements that are likely part of the same heading"""
        for page_data in pages_data:
            merged_elements = []
            i = 0
            
            while i < len(page_data["text_elements"]):
                current = page_data["text_elements"][i]
                
                # Check if this might be part of a heading that continues
                if i + 1 < len(page_data["text_elements"]):
                    next_elem = page_data["text_elements"][i+1]
                    
                    # If elements are close vertically and have similar properties
                    if (abs(next_elem["y0"] - current["y1"]) < 10 and
                        abs(current["size"] - next_elem["size"]) < 2 and
                        current["flags"] == next_elem["flags"] and
                        len(current["text"]) + len(next_elem["text"]) < 100):
                        
                        # Merge the elements
                        merged = current.copy()
                        merged["text"] = current["text"] + " " + next_elem["text"]
                        merged["bbox"] = [
                            min(current["bbox"][0], next_elem["bbox"][0]),
                            min(current["bbox"][1], next_elem["bbox"][1]),
                            max(current["bbox"][2], next_elem["bbox"][2]),
                            max(current["bbox"][3], next_elem["bbox"][3])
                        ]
                        merged["x0"] = merged["bbox"][0]
                        merged["y0"] = merged["bbox"][1]
                        merged["x1"] = merged["bbox"][2]
                        merged["y1"] = merged["bbox"][3]
                        
                        merged_elements.append(merged)
                        i += 2
                        continue
                
                merged_elements.append(current)
                i += 1
            
            page_data["text_elements"] = merged_elements
            
        return pages_data
    
    def is_heading_candidate(self, element: Dict) -> bool:
        """Determine if text element could be a heading"""
        text = element["text"].strip()
        
        # Skip empty or very short text
        if len(text) < 2:
            return False
        
        # Skip very long text (likely paragraphs)
        if len(text) > 300:
            return False
        
        # Check font size (should be larger than body text)
        font_size = element["size"]
        if font_size < 8:
            return False
        
        # Check if text has heading-like characteristics
        is_bold = element["flags"] & 16  # Bold flag
        is_title_case = text.istitle() or text.isupper()
        
        # Check for section numbering patterns
        has_section_numbering = any(re.match(pattern, text) for pattern in self.section_patterns)
        
        # Check for heading terms
        contains_heading_term = any(term in text.lower() for term in self.heading_terms)
        
        # Check if text ends with colon (often indicates a heading)
        ends_with_colon = text.endswith(':')
        
        # Check if text is short (headings tend to be shorter)
        is_short = len(text) < 100
        
        # Basic heuristics for heading detection
        return (
            # Font characteristics
            (font_size > 12 and (is_bold or is_title_case)) or
            # Structural characteristics
            has_section_numbering or
            # Content characteristics
            (contains_heading_term and is_short) or
            # Short bold text is likely a heading
            (is_bold and is_short and not text.endswith('.')) or
            # Colon endings often indicate headings
            (ends_with_colon and is_short)
        )
    
    def get_layout_features(self, element: Dict, page_data: Dict) -> Dict:
        """Extract layout features for heading classification"""
        bbox = element["bbox"]
        page_width = page_data["page_width"]
        page_height = page_data["page_height"]
        
        # Normalize coordinates
        x_center = (bbox[0] + bbox[2]) / (2 * page_width)
        y_center = (bbox[1] + bbox[3]) / (2 * page_height)
        
        # Calculate relative font size
        sizes = [e["size"] for e in page_data["text_elements"] if e["size"] > 0]
        avg_font_size = np.mean(sizes) if sizes else 12
        font_ratio = element["size"] / max(avg_font_size, 1)
        
        # Calculate text density
        text_length = len(element["text"])
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        text_density = text_length / max(bbox_area, 1) * 1000  # Scaled for readability
        
        # Check if element is at the start of a new section
        is_at_top = bbox[1] < page_height * 0.2
        
        # Check if element is indented
        is_indented = bbox[0] > page_width * 0.1
        
        # Check for section numbering
        has_section_numbering = any(re.match(pattern, element["text"]) for pattern in self.section_patterns)
        
        # Check for heading terms
        contains_heading_term = any(term in element["text"].lower() for term in self.heading_terms)
        
        # Check if text is all caps (often indicates heading)
        is_all_caps = element["text"].isupper() and len(element["text"]) > 3
        
        # Check if text is title case
        is_title_case = element["text"].istitle() and len(element["text"]) > 3
        
        return {
            "x_center": x_center,
            "y_center": y_center,
            "font_ratio": font_ratio,
            "is_bold": bool(element["flags"] & 16),
            "is_italic": bool(element["flags"] & 2),
            "text_length": text_length,
            "text_density": text_density,
            "is_at_top": is_at_top,
            "is_indented": is_indented,
            "has_section_numbering": has_section_numbering,
            "contains_heading_term": contains_heading_term,
            "is_all_caps": is_all_caps,
            "is_title_case": is_title_case,
            "font_size": element["size"],
            "page": element["page"]
        }
    
    def _clean_heading_text(self, text: str) -> str:
        """Clean and normalize heading text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common section numbering
        for pattern in self.section_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove trailing punctuation
        text = re.sub(r'[.,:;]$', '', text)
        
        # Limit length
        if len(text) > 150:
            text = text[:147] + "..."
        
        return text.strip() 