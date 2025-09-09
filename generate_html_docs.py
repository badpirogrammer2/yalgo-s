#!/usr/bin/env python3
"""
Script to generate HTML documentation from Markdown files
"""

import os
import markdown
from pathlib import Path

def markdown_to_html(markdown_content, title="YALGO-S Documentation"):
    """Convert markdown content to HTML with professional styling."""
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - YALGO-S</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
        }}

        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            color: #2980b9;
        }}

        h2 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }}

        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
        }}

        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }}

        pre code {{
            background: none;
            padding: 0;
        }}

        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}

        tr:hover {{
            background-color: #f8f9fa;
        }}

        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}

        a {{
            color: #3498db;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }}

        .badge-success {{
            background-color: #d4edda;
            color: #155724;
        }}

        .badge-info {{
            background-color: #d1ecf1;
            color: #0c5460;
        }}

        ul, ol {{
            padding-left: 30px;
        }}

        li {{
            margin-bottom: 5px;
        }}

        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header style="text-align: center; margin-bottom: 40px;">
            <h1>🚀 YALGO-S Documentation</h1>
            <p>Advanced AI Algorithms for Optimization, Multi-Modal Processing, and Adaptive Learning</p>
        </header>

        <nav style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 30px;">
            <strong>Quick Navigation:</strong>
            <a href="#top">Top</a> |
            <a href="README.html">Main README</a> |
            <a href="ALGOs/New%20Algos/Readme.html">Installation</a> |
            <a href="ALGOs/New%20Algos/applications.html">Applications</a>
        </nav>

        <main>
"""

    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code', 'codehilite'])

    # Close the HTML template
    html_template += html_content
    html_template += """
        </main>

        <div class="footer">
            <p>
                <strong>YALGO-S</strong> - Advanced AI Algorithms<br>
                <a href="https://github.com/badpirogrammer2/yalgo-s">GitHub Repository</a> |
                <a href="https://docs.yalgo-s.com">Official Documentation</a>
            </p>
        </div>
    </div>
</body>
</html>"""

    return html_template

def generate_html_from_markdown():
    """Generate HTML files from Markdown files."""

    print("🚀 Generating HTML Documentation from Markdown Files")
    print("=" * 60)

    # Define markdown files to convert
    markdown_files = {
        'docs/installation.md': 'docs/installation.html',
        'docs/quickstart.md': 'docs/quickstart.html',
        'docs/best-practices.md': 'docs/best-practices.html',
        'docs/development.md': 'docs/development.html',
        'ALGOs/New Algos/AGMOHD/readme.md': 'ALGOs/New Algos/AGMOHD/readme.html'
    }

    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)

    generated_files = []

    for md_file, html_file in markdown_files.items():
        if os.path.exists(md_file):
            print(f"📄 Converting: {md_file} → {html_file}")

            try:
                # Read markdown content
                with open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()

                # Extract title from first heading
                lines = markdown_content.split('\n')
                title = "YALGO-S Documentation"
                for line in lines[:10]:  # Check first 10 lines
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break

                # Convert to HTML
                html_content = markdown_to_html(markdown_content, title)

                # Write HTML file
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                generated_files.append(html_file)
                print(f"✅ Generated: {html_file}")

            except Exception as e:
                print(f"❌ Error converting {md_file}: {str(e)}")
        else:
            print(f"⚠️  Source file not found: {md_file}")

    print("\n" + "=" * 60)
    print("📁 HTML Generation Summary:")
    print("=" * 60)

    if generated_files:
        print(f"✅ Successfully generated {len(generated_files)} HTML files:")
        for file in generated_files:
            print(f"   • {file}")
    else:
        print("❌ No HTML files were generated")

    print("\n🎉 HTML Documentation Generation Complete!")
    print("💡 You can now open these HTML files directly in your web browser")

if __name__ == "__main__":
    generate_html_from_markdown()
