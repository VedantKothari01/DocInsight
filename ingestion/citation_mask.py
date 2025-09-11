"""
Citation masking module for DocInsight

Provides improved citation detection and masking capabilities for academic documents.
Supports numeric, author-year, and footnote citation formats with toggle functionality.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import CITATION_PATTERNS, CITATION_MASKING_ENABLED

logger = logging.getLogger(__name__)


@dataclass
class CitationMatch:
    """Represents a detected citation in text"""
    pattern_type: str
    original_text: str
    start_pos: int
    end_pos: int
    replacement: str


class CitationMasker:
    """Handles citation detection and masking in academic documents"""
    
    def __init__(self, enabled: bool = None):
        """Initialize citation masker
        
        Args:
            enabled: Whether citation masking is enabled. If None, uses config default.
        """
        self.enabled = enabled if enabled is not None else CITATION_MASKING_ENABLED
        self.citation_counter = 0
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for pattern_type, patterns in CITATION_PATTERNS.items():
            self.compiled_patterns[pattern_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def detect_citations(self, text: str) -> List[CitationMatch]:
        """Detect all citations in text
        
        Args:
            text: Input text to scan for citations
            
        Returns:
            List of CitationMatch objects sorted by position
        """
        if not self.enabled:
            return []
        
        citations = []
        
        for pattern_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    self.citation_counter += 1
                    citations.append(CitationMatch(
                        pattern_type=pattern_type,
                        original_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        replacement=f"[CITE_{self.citation_counter}]"
                    ))
        
        # Sort by position to process replacements correctly
        citations.sort(key=lambda x: x.start_pos)
        return citations
    
    def mask_citations(self, text: str, keep_structure: bool = True) -> Tuple[str, List[CitationMatch]]:
        """Mask citations in text while preserving document structure
        
        Args:
            text: Input text containing citations
            keep_structure: Whether to preserve citation structure with placeholders
            
        Returns:
            Tuple of (masked_text, list_of_citations)
        """
        if not self.enabled:
            return text, []
        
        citations = self.detect_citations(text)
        if not citations:
            return text, []
        
        # Apply replacements from right to left to preserve positions
        masked_text = text
        for citation in reversed(citations):
            if keep_structure:
                # Replace with placeholder token
                masked_text = (
                    masked_text[:citation.start_pos] + 
                    citation.replacement + 
                    masked_text[citation.end_pos:]
                )
            else:
                # Remove citation entirely
                masked_text = (
                    masked_text[:citation.start_pos] + 
                    masked_text[citation.end_pos:]
                )
        
        logger.debug(f"Masked {len(citations)} citations in text")
        return masked_text, citations
    
    def unmask_citations(self, masked_text: str, citations: List[CitationMatch]) -> str:
        """Restore original citations in masked text
        
        Args:
            masked_text: Text with citation placeholders
            citations: List of original citations to restore
            
        Returns:
            Text with original citations restored
        """
        if not citations:
            return masked_text
        
        unmasked_text = masked_text
        
        # Replace placeholders with original citations
        for citation in citations:
            unmasked_text = unmasked_text.replace(
                citation.replacement, 
                citation.original_text
            )
        
        return unmasked_text
    
    def get_citation_summary(self, citations: List[CitationMatch]) -> Dict[str, int]:
        """Get summary statistics for detected citations
        
        Args:
            citations: List of detected citations
            
        Returns:
            Dictionary with citation counts by type
        """
        summary = {pattern_type: 0 for pattern_type in CITATION_PATTERNS.keys()}
        summary['total'] = len(citations)
        
        for citation in citations:
            summary[citation.pattern_type] += 1
        
        return summary


def mask_citations_in_text(text: str, enabled: bool = None) -> Tuple[str, str, Dict]:
    """Convenience function to mask citations in text
    
    Args:
        text: Input text
        enabled: Whether masking is enabled
        
    Returns:
        Tuple of (masked_text, original_text, citation_info)
    """
    masker = CitationMasker(enabled=enabled)
    masked_text, citations = masker.mask_citations(text)
    citation_info = masker.get_citation_summary(citations)
    
    return masked_text, text, citation_info


def get_citation_stats(text: str) -> Dict[str, int]:
    """Get citation statistics without masking
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with citation counts by type
    """
    masker = CitationMasker(enabled=True)
    citations = masker.detect_citations(text)
    return masker.get_citation_summary(citations)


# Example usage for testing
if __name__ == "__main__":
    sample_text = """
    Previous research has shown significant results [1, 2]. Smith et al. (2020) 
    demonstrated this approach. According to recent studies (Johnson, 2019), 
    the methodology is sound. See footnote 1) for details.
    """
    
    masker = CitationMasker()
    masked, citations = masker.mask_citations(sample_text)
    print("Original:", sample_text)
    print("Masked:", masked)
    print("Citations found:", masker.get_citation_summary(citations))