#!/usr/bin/env python3
"""
Generate a PDF document containing all Python source files
in the directory of this script.

This version safely handles non‑ASCII characters (e.g., Persian comments)
by replacing them with '?' so that the built‑in Latin‑1 fonts work.

Usage:
    python generate_code_pdf.py

Output:
    phase10_source_code.pdf
"""

import os
import sys
import glob
from datetime import datetime
from fpdf import FPDF

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
OUTPUT_PDF = "phase10_source_code.pdf"
FONT_NAME = "Courier"          # Monospace for code
FONT_SIZE = 8                   # 8pt is good for compact listing
HEADER_FONT_SIZE = 12
LINE_SPACING = 4.5              # mm between lines
MARGIN = 15                     # page margin in mm
PAGE_BREAK_MARGIN = 20         # mm from bottom to trigger page break

# ----------------------------------------------------------------------
# Helper to sanitize text for Latin-1 fonts
# ----------------------------------------------------------------------
def safe_text(text):
    """
    Replace all characters that are not in the Latin-1 range with '?'.
    This prevents fpdf2's built-in fonts from raising Unicode errors.
    """
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        # Replace each character that is > 255 with '?'
        return ''.join(ch if ord(ch) < 256 else '?' for ch in text)

# ----------------------------------------------------------------------
# PDF Class (customized for code listing)
# ----------------------------------------------------------------------
class CodePDF(FPDF):
    def __init__(self):
        super().__init__(format='A4', unit='mm')
        self.set_auto_page_break(auto=True, margin=PAGE_BREAK_MARGIN)
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_title("Phase 10 - Source Code Documentation")

    def header(self):
        self.set_font(FONT_NAME, 'B', HEADER_FONT_SIZE)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, safe_text("Phase 10 - Source Code Documentation"), align='C')
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(MARGIN, self.get_y(), self.w - MARGIN, self.get_y())

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_NAME, 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, safe_text(f"Page {self.page_no()}"), align='C')

    # Add a file heading
    def add_file_heading(self, filename, file_number):
        self.set_font(FONT_NAME, 'B', 10)
        self.set_text_color(0, 0, 120)
        self.cell(0, 6, safe_text(f"File {file_number}: {filename}"), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_text_color(80, 80, 80)

    # Add code content with basic syntax coloring
    def add_code_block(self, code_lines):
        self.set_font(FONT_NAME, '', FONT_SIZE)
        self.set_text_color(0, 0, 0)
        for line in code_lines:
            # Sanitize the line before printing
            safe_line = safe_text(line.replace('\t', '    '))
            
            # Basic colouring based on leading characters
            stripped = safe_line.strip()
            if stripped.startswith('#'):
                self.set_text_color(128, 128, 128)          # grey for comments
            elif stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith('//'):
                self.set_text_color(128, 128, 128)
            elif any(keyword in safe_line for keyword in ['def ', 'class ', 'import ', 'from ']):
                self.set_text_color(0, 0, 180)              # blue for declarations
            elif any(keyword in safe_line for keyword in ['return', 'if ', 'for ', 'while ', 'except ', 'try:']):
                self.set_text_color(150, 0, 0)              # red for control flow
            else:
                self.set_text_color(0, 0, 0)                # normal black
            
            self.cell(0, LINE_SPACING, safe_line, new_x="LMARGIN", new_y="NEXT")
            if self.get_y() > self.h - PAGE_BREAK_MARGIN:
                self.add_page()

# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------
def main():
    # Determine the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Find all .py files in that directory
    py_files = sorted(glob.glob(os.path.join(script_dir, "*.py")))
    
    # Remove the script itself from the list
    script_name = os.path.basename(__file__)
    py_files = [f for f in py_files if os.path.basename(f) != script_name]
    
    if not py_files:
        print("No .py files found in the script's directory.")
        return

    pdf = CodePDF()
    pdf.add_page()

    # Title page
    pdf.set_font(FONT_NAME, 'B', 24)
    pdf.set_text_color(0, 0, 80)
    pdf.cell(0, 20, safe_text("Phase 10"), align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(FONT_NAME, '', 16)
    pdf.cell(0, 10, safe_text("Source Code Documentation"), align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font(FONT_NAME, '', 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, safe_text(f"Total Files: {len(py_files)}"), align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, safe_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)

    # Table of contents
    pdf.set_font(FONT_NAME, 'B', 14)
    pdf.set_text_color(0, 0, 100)
    pdf.cell(0, 8, safe_text("Table of Contents"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font(FONT_NAME, '', 10)
    pdf.set_text_color(0, 0, 0)
    for i, fname in enumerate(py_files, 1):
        # Use just the file name for display
        display_name = os.path.basename(fname)
        pdf.cell(0, LINE_SPACING + 1, safe_text(f"{i:3d}. {display_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.add_page()

    # Process each file
    for idx, filename in enumerate(py_files, 1):
        print(f"Processing: {os.path.basename(filename)} (file {idx} of {len(py_files)})")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

        lines = file_content.splitlines()
        pdf.add_file_heading(os.path.basename(filename), idx)
        pdf.add_code_block(lines)
        
        if idx < len(py_files):
            pdf.add_page()
    
    # Save the PDF in the same directory as the script
    output_path = os.path.join(script_dir, OUTPUT_PDF)
    pdf.output(output_path)
    print(f"\nPDF generated successfully: {output_path}")
    print(f"Total pages: {pdf.page_no()}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()