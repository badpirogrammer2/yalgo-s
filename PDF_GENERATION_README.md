# YALGO-S PDF Documentation

## 📄 Available Documentation Formats

YALGO-S documentation is available in multiple formats for your convenience:

### 🌐 HTML Files (Ready to View)
- `README.html` - Main project documentation
- `ALGOs/New Algos/applications.html` - Applications and use cases guide
- `ALGOs/New Algos/Readme.html` - Installation and usage guide

### 📖 Markdown Files (Source)
- `README.md` - Main project documentation
- `ALGOs/New Algos/applications.md` - Applications and use cases guide
- `ALGOs/New Algos/Readme` - Installation and usage guide

## 🖨️ Converting HTML to PDF

### Method 1: Browser Print (Recommended)
1. Open any HTML file in your web browser (Chrome, Firefox, Safari, Edge)
2. Press `Ctrl+P` (or `Cmd+P` on Mac) to open print dialog
3. Select "Save as PDF" or "Print to PDF"
4. Choose paper size (A4 recommended)
5. Save the PDF file

### Method 2: Online Converters
Use any of these free online HTML to PDF converters:
- `https://html2pdf.com/`
- `https://www.pdfcrowd.com/html-to-pdf/`
- `https://www.sejda.com/html-to-pdf`

### Method 3: Command Line Tools

#### Using wkhtmltopdf (if available)
```bash
# Install wkhtmltopdf first
# Then convert HTML files
wkhtmltopdf README.html README.pdf
wkhtmltopdf ALGOs/New\ Algos/applications.html applications.pdf
wkhtmltopdf ALGOs/New\ Algos/Readme.html Readme.pdf
```

#### Using pandoc (if available)
```bash
# Convert HTML to PDF via LaTeX
pandoc README.html -o README.pdf
pandoc ALGOs/New\ Algos/applications.html -o applications.pdf
pandoc ALGOs/New\ Algos/Readme.html -o Readme.pdf
```

### Method 4: Python Libraries

#### Using WeasyPrint (requires system libraries)
```bash
pip install weasyprint
python -c "import weasyprint; weasyprint.HTML('README.html').write_pdf('README.pdf')"
```

#### Using pdfkit (requires wkhtmltopdf)
```bash
pip install pdfkit
python -c "import pdfkit; pdfkit.from_file('README.html', 'README.pdf')"
```

## 📋 Documentation Contents

### README.html / README.pdf
- Project overview and features
- Installation instructions
- AGMOHD, POIC-NET, ARCE, and Image Training examples
- Benchmark results and performance metrics
- Real-world applications
- Cross-platform compatibility information

### applications.html / applications.pdf
- Detailed applications for each algorithm
- Performance metrics and benchmarks
- Usage examples and code samples
- Combined algorithm applications
- Future enhancements roadmap

### Readme.html / Readme.pdf
- Complete installation guide for all platforms
- System requirements and dependencies
- Testing and validation procedures
- Troubleshooting common issues
- Performance optimization tips

## 🎯 Key Features Documented

### ✅ Image Training Functionality (NEW)
- **Easy-to-Use API**: Simple interface for training CNNs
- **AGMOHD Integration**: Advanced optimization with hindrance detection
- **Pre-trained Models**: Support for ResNet, VGG, AlexNet architectures
- **Data Augmentation**: Built-in augmentation for better generalization
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Dataset Support**: CIFAR-10, MNIST, and custom datasets

### ✅ Complete Algorithm Suite
- **AGMOHD**: Adaptive Gradient Momentum with Hindrance Detection
- **POIC-NET**: Partial Object Inference and Completion Network
- **ARCE**: Adaptive Resonance with Contextual Embedding
- **Image Training**: Integrated CNN training with AGMOHD optimizer

### ✅ Comprehensive Documentation
- Installation guides for all platforms (Linux, macOS, Windows)
- Performance benchmarks and optimization tips
- Real-world applications and use cases
- Troubleshooting and support information

## 📁 File Structure

```
YALGO-S/
├── README.html                    # Main documentation (HTML)
├── README.md                      # Main documentation (Markdown)
├── ALGOs/New Algos/
│   ├── applications.html          # Applications guide (HTML)
│   ├── applications.md            # Applications guide (Markdown)
│   ├── Readme.html               # Installation guide (HTML)
│   └── Readme                    # Installation guide (Markdown)
└── docs/pdf/                      # PDF versions (when generated)
    ├── README.pdf
    ├── applications.pdf
    └── Readme.pdf
```

## 🚀 Quick Start

1. **View HTML**: Open any `.html` file in your web browser
2. **Generate PDF**: Use browser print or online converters
3. **Read Documentation**: All files contain the same comprehensive information

## 💡 Tips

- **HTML files** are best for online viewing and navigation
- **PDF files** are ideal for printing and offline reading
- **Markdown files** are for developers who want to contribute
- All formats contain identical content with the latest Image Training functionality

## 📞 Support

For questions about documentation or PDF generation:
- Check the troubleshooting section in the documentation
- Visit the GitHub repository for issues and discussions
- All documentation includes contact information and support links

---

**Last Updated**: September 8, 2025
**YALGO-S Version**: 0.1.0
**Includes**: Complete Image Training functionality documentation
