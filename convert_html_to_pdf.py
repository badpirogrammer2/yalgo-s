#!/usr/bin/env python3
"""
Script to convert HTML documentation files to PDF format using WeasyPrint
"""

import weasyprint
import os
from pathlib import Path

def convert_html_to_pdf():
    """Convert HTML files to PDF format using WeasyPrint"""

    # Define the HTML files to convert
    html_files = [
        'README.html',
        'ALGOs/New Algos/applications.html',
        'ALGOs/New Algos/Readme.html'
    ]

    # PDF output directory
    pdf_dir = 'docs/pdf'
    os.makedirs(pdf_dir, exist_ok=True)

    print("🚀 Converting HTML files to PDF using WeasyPrint...")
    print("=" * 60)

    for html_file in html_files:
        if os.path.exists(html_file):
            # Generate PDF filename
            pdf_filename = Path(html_file).stem + '.pdf'
            pdf_path = os.path.join(pdf_dir, pdf_filename)

            try:
                print(f"📄 Converting {html_file} → {pdf_path}")

                # Convert HTML to PDF using WeasyPrint
                weasyprint.HTML(html_file).write_pdf(pdf_path)

                # Check if PDF was created successfully
                if os.path.exists(pdf_path):
                    file_size = os.path.getsize(pdf_path)
                    print(f"✅ Successfully created {pdf_path} ({file_size:,} bytes)")
                else:
                    print(f"❌ Failed to create {pdf_path}")

            except Exception as e:
                print(f"❌ Error converting {html_file}: {str(e)}")
        else:
            print(f"⚠️  HTML file not found: {html_file}")

    print("\n" + "=" * 60)
    print("📋 PDF Generation Summary:")
    print("=" * 60)

    # List all generated PDFs
    if os.path.exists(pdf_dir):
        pdf_files = list(Path(pdf_dir).glob('*.pdf'))
        if pdf_files:
            print(f"📁 Generated {len(pdf_files)} PDF files in {pdf_dir}/:")
            for pdf_file in pdf_files:
                size = os.path.getsize(pdf_file)
                print(f"   • {pdf_file.name} ({size:,} bytes)")
        else:
            print("❌ No PDF files were generated")
    else:
        print("❌ PDF directory was not created")

    print("\n💡 PDF files can be opened with any PDF reader")
    print("📖 They contain the complete YALGO-S documentation with Image Training functionality")
    print("🎨 Generated using WeasyPrint - pure Python HTML to PDF converter")

if __name__ == "__main__":
    convert_html_to_pdf()
