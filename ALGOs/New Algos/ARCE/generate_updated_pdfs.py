#!/usr/bin/env python3
"""
Script to generate updated PDF documentation for ARCE and YALGO-S
"""

import os
import subprocess
from pathlib import Path

def generate_updated_pdfs():
    """Generate updated PDF files with new Image Training functionality"""

    print("🚀 Generating Updated PDF Documentation for YALGO-S")
    print("=" * 60)

    # Define source files and target PDFs
    pdf_sources = {
        'ARCE_Complete_Documentation.md': 'ARCE_Complete_Documentation.pdf',
        '../applications.md': 'applications_updated.pdf',
        '../Readme': 'installation_guide_updated.pdf'
    }

    # Create output directory
    pdf_dir = 'updated_pdfs'
    os.makedirs(pdf_dir, exist_ok=True)

    print("📄 Available conversion methods:")
    print("1. Pandoc (recommended)")
    print("2. Browser print")
    print("3. Online converters")
    print("4. Python libraries")
    print()

    for source_file, target_pdf in pdf_sources.items():
        if os.path.exists(source_file):
            pdf_path = os.path.join(pdf_dir, target_pdf)
            print(f"📋 Processing: {source_file} → {pdf_path}")

            # Try pandoc first
            try:
                cmd = ['pandoc', source_file, '-o', pdf_path, '--pdf-engine=pdflatex']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    file_size = os.path.getsize(pdf_path)
                    print(f"✅ Generated with pandoc: {file_size:,} bytes")
                else:
                    print(f"⚠️  Pandoc failed: {result.stderr[:100]}...")
                    print("   Try: brew install pandoc texlive")
                    print(f"   Then: pandoc {source_file} -o {pdf_path}")

            except (subprocess.TimeoutExpired, FileNotFoundError):
                print("   Pandoc not available - trying alternative methods...")

            # Provide alternative instructions
            print(f"   Alternative: Open {source_file} in browser and print to PDF")
            print(f"   Online: Use html2pdf.com or similar service")
            print()

        else:
            print(f"❌ Source file not found: {source_file}")

    print("=" * 60)
    print("📁 PDF Generation Summary:")
    print("=" * 60)

    # Check generated files
    if os.path.exists(pdf_dir):
        pdf_files = list(Path(pdf_dir).glob('*.pdf'))
        if pdf_files:
            print(f"📂 Generated {len(pdf_files)} PDF files:")
            for pdf_file in pdf_files:
                size = os.path.getsize(pdf_file)
                print(f"   • {pdf_file.name} ({size:,} bytes)")
        else:
            print("❌ No PDF files were generated")
    else:
        print("❌ PDF directory was not created")

    print()
    print("🔧 Manual PDF Generation Instructions:")
    print("=" * 60)
    print("1. Install pandoc and LaTeX:")
    print("   brew install pandoc texlive")
    print()
    print("2. Convert Markdown to PDF:")
    print("   pandoc ARCE_Complete_Documentation.md -o ARCE_Complete_Documentation.pdf")
    print("   pandoc ../applications.md -o applications_updated.pdf")
    print("   pandoc ../Readme -o installation_guide_updated.pdf")
    print()
    print("3. Alternative - Browser method:")
    print("   • Open Markdown file in GitHub/GitLab")
    print("   • Use browser's Print → Save as PDF")
    print("   • Or use online Markdown to PDF converters")
    print()
    print("4. Alternative - Python method:")
    print("   pip install markdown-pdf")
    print("   markdown-pdf ARCE_Complete_Documentation.md")
    print()
    print("📖 The updated documentation includes:")
    print("   • Complete ARCE algorithm documentation")
    print("   • New Image Training functionality")
    print("   • AGMOHD optimizer integration")
    print("   • Performance benchmarks")
    print("   • Usage examples and applications")

if __name__ == "__main__":
    generate_updated_pdfs()
