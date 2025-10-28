"""
Section parser module for DocInsight

Provides academic document structure detection and segmentation into logical sections.
Supports section detection through heading patterns and fallback to rolling window approach.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import ACADEMIC_SECTIONS, SECTION_MIN_TOKENS, CHUNK_SIZE, OVERLAP

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section in an academic document"""
    section_type: str
    title: str
    content: str
    start_pos: int
    end_pos: int
    token_count: int
    confidence: float  # Confidence in section classification


class AcademicSectionParser:
    """Parses academic documents into logical sections"""
    
    def __init__(self, min_section_tokens: int = None):
        """Initialize section parser
        
        Args:
            min_section_tokens: Minimum tokens required for a valid section
        """
        self.min_section_tokens = min_section_tokens or SECTION_MIN_TOKENS
        
        # Compile section heading patterns
        self.section_patterns = self._compile_section_patterns()
    
    def _compile_section_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for academic section detection"""
        patterns = {}
        
        for section_type, keywords in ACADEMIC_SECTIONS.items():
            patterns[section_type] = []
            for keyword in keywords:
                # Pattern for section headings (case insensitive)
                # Matches: "1. Introduction", "I. METHODS", "Abstract", etc.
                pattern = re.compile(
                    rf'^[\s]*(?:\d+\.?\s*|\w+\.?\s*)?{re.escape(keyword)}[\s]*:?[\s]*$',
                    re.IGNORECASE | re.MULTILINE
                )
                patterns[section_type].append(pattern)
        
        return patterns
    
    def detect_section_headings(self, text: str) -> List[Tuple[str, int, int, float]]:
        """Detect section headings in text
        
        Args:
            text: Document text to analyze
            
        Returns:
            List of tuples: (section_type, start_pos, end_pos, confidence)
        """
        headings = []
        
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    confidence = 0.8  # Base confidence for regex match
                    
                    # Increase confidence if heading is at start of line
                    if match.start() == 0 or text[match.start() - 1] == '\n':
                        confidence += 0.1
                    
                    # Increase confidence if followed by content
                    remaining_text = text[match.end():match.end() + 100]
                    if remaining_text.strip():
                        confidence += 0.1
                    
                    headings.append((
                        section_type,
                        match.start(),
                        match.end(),
                        min(confidence, 1.0)
                    ))
        
        # Sort by position and remove overlaps
        headings.sort(key=lambda x: x[1])
        return self._remove_overlapping_headings(headings)
    
    def _remove_overlapping_headings(self, headings: List[Tuple[str, int, int, float]]) -> List[Tuple[str, int, int, float]]:
        """Remove overlapping section headings, keeping highest confidence"""
        if not headings:
            return []
        
        filtered = [headings[0]]
        
        for current in headings[1:]:
            last = filtered[-1]
            
            # Check if headings overlap
            if current[1] < last[2]:
                # Keep heading with higher confidence
                if current[3] > last[3]:
                    filtered[-1] = current
            else:
                filtered.append(current)
        
        return filtered
    
    def parse_sections(self, text: str) -> List[DocumentSection]:
        """Parse document into academic sections
        
        Args:
            text: Document text to parse
            
        Returns:
            List of DocumentSection objects
        """
        if not text.strip():
            return []
        
        headings = self.detect_section_headings(text)
        
        if not headings:
            logger.info("No section headings found, using fallback segmentation")
            return self._fallback_segmentation(text)
        
        sections = []
        
        for i, (section_type, start, end, confidence) in enumerate(headings):
            # Determine content boundaries
            content_start = end
            content_end = headings[i + 1][1] if i + 1 < len(headings) else len(text)
            
            # Extract title and content
            title = text[start:end].strip()
            content = text[content_start:content_end].strip()
            
            # Count tokens (approximate)
            token_count = len(content.split())
            
            # Skip sections that are too short
            if token_count < self.min_section_tokens:
                logger.debug(f"Skipping short section '{title}' ({token_count} tokens)")
                continue
            
            sections.append(DocumentSection(
                section_type=section_type,
                title=title,
                content=content,
                start_pos=start,
                end_pos=content_end,
                token_count=token_count,
                confidence=confidence
            ))
        
        logger.info(f"Parsed {len(sections)} sections from document")
        return sections
    
    def _fallback_segmentation(self, text: str) -> List[DocumentSection]:
        """Fallback segmentation using rolling window approach
        
        Args:
            text: Document text to segment
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        text_tokens = text.split()
        
        if len(text_tokens) < self.min_section_tokens:
            # Document too short for segmentation
            return [DocumentSection(
                section_type='unknown',
                title='Document',
                content=text,
                start_pos=0,
                end_pos=len(text),
                token_count=len(text_tokens),
                confidence=0.5
            )]
        
        # Create segments of CHUNK_SIZE tokens with OVERLAP
        i = 0
        section_num = 1
        
        while i < len(text_tokens):
            end_idx = min(i + CHUNK_SIZE, len(text_tokens))
            segment_tokens = text_tokens[i:end_idx]
            segment_text = ' '.join(segment_tokens)
            
            # Find character positions
            chars_before = len(' '.join(text_tokens[:i]))
            start_pos = chars_before + (1 if chars_before > 0 else 0)
            end_pos = start_pos + len(segment_text)
            
            sections.append(DocumentSection(
                section_type='segment',
                title=f'Segment {section_num}',
                content=segment_text,
                start_pos=start_pos,
                end_pos=end_pos,
                token_count=len(segment_tokens),
                confidence=0.3  # Low confidence for automatic segmentation
            ))
            
            section_num += 1
            i += CHUNK_SIZE - OVERLAP
        
        logger.info(f"Created {len(sections)} fallback segments")
        return sections
    
    def get_section_summary(self, sections: List[DocumentSection]) -> Dict[str, any]:
        """Get summary statistics for parsed sections
        
        Args:
            sections: List of parsed sections
            
        Returns:
            Dictionary with section statistics
        """
        if not sections:
            return {'total_sections': 0, 'total_tokens': 0, 'section_types': {}}
        
        total_tokens = sum(section.token_count for section in sections)
        section_types = {}
        
        for section in sections:
            if section.section_type not in section_types:
                section_types[section.section_type] = 0
            section_types[section.section_type] += 1
        
        avg_confidence = sum(section.confidence for section in sections) / len(sections)
        
        return {
            'total_sections': len(sections),
            'total_tokens': total_tokens,
            'section_types': section_types,
            'avg_confidence': avg_confidence,
            'has_academic_structure': any(
                section.section_type in ACADEMIC_SECTIONS 
                for section in sections
            )
        }


def parse_academic_document(text: str) -> Tuple[List[DocumentSection], Dict[str, any]]:
    """Convenience function to parse academic document
    
    Args:
        text: Document text to parse
        
    Returns:
        Tuple of (sections, summary_stats)
    """
    parser = AcademicSectionParser()
    sections = parser.parse_sections(text)
    summary = parser.get_section_summary(sections)
    
    return sections, summary


# Example usage for testing
if __name__ == "__main__":
    sample_text = """
    Abstract
    This paper presents a novel approach to document analysis.
    
    1. Introduction
    Document analysis is a critical task in modern information systems.
    Previous work has focused on simple keyword matching.
    
    2. Methods
    We propose a new algorithm based on machine learning techniques.
    The approach uses semantic embeddings to understand content.
    
    3. Results
    Our experiments show significant improvements over baseline methods.
    The accuracy increased by 25% on average across all test cases.
    
    4. Conclusion
    This work demonstrates the effectiveness of semantic approaches.
    Future work will explore cross-lingual applications.
    
    References
    [1] Smith, J. (2020). Document Analysis Methods.
    [2] Jones, A. (2019). Semantic Processing Techniques.
    """
    
    sections, summary = parse_academic_document(sample_text)
    print(f"Found {len(sections)} sections:")
    for section in sections:
        print(f"- {section.section_type}: {section.title} ({section.token_count} tokens)")
    print(f"Summary: {summary}")