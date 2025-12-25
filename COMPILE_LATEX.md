# LaTeX Document Compilation Guide

Your evaluation results have been saved as **`evaluation_results.tex`**

## Option 1: Compile on Your Local Machine

If you have LaTeX installed locally:

```bash
# On Windows (MiKTeX or TexLive)
pdflatex evaluation_results.tex
pdflatex evaluation_results.tex  # Run twice for TOC

# On macOS (MacTeX)
pdflatex evaluation_results.tex
pdflatex evaluation_results.tex

# On Linux (TexLive)
sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-latex-extra
pdflatex evaluation_results.tex
pdflatex evaluation_results.tex
```

## Option 2: Use Online LaTeX Compilers

These free services compile LaTeX online:

1. **Overleaf** (https://www.overleaf.com)
   - Upload `evaluation_results.tex`
   - Download PDF
   - No installation needed

2. **pdflatex Online** (https://www.writelatex.com)
   - Similar to Overleaf

3. **LaTeX Compile Service** (https://www.latex4j.com)
   - Quick online compilation

## Option 3: Docker Compilation

If you have Docker installed:

```bash
docker run --rm -v /home/vasanthiyer-gpu:/data \
  blang/latex:ubuntu \
  pdflatex -interaction=nonstopmode /data/evaluation_results.tex

docker run --rm -v /home/vasanthiyer-gpu:/data \
  blang/latex:ubuntu \
  pdflatex -interaction=nonstopmode /data/evaluation_results.tex
```

## Document Contents

Your LaTeX document includes:

✅ Executive Summary
✅ Evaluation Methodology
✅ Results Summary Table
✅ Score Distribution Charts
✅ Detailed Query Analysis
✅ Performance Analysis
✅ Latency Metrics
✅ Entity Coverage
✅ Recommendations
✅ Technical Specifications
✅ Database Statistics
✅ Appendices with Scoring Rubric

## Output Files Expected

After compilation:
- `evaluation_results.pdf` - Main report
- `evaluation_results.aux` - Auxiliary file
- `evaluation_results.log` - Compilation log
- `evaluation_results.toc` - Table of contents

## Document Statistics

- **Pages:** ~12
- **Tables:** 12
- **Figures:** 2 (with PGFPlots)
- **Sections:** 10
- **Appendices:** 3

## If You Want HTML Instead

To convert to HTML/web format:

```bash
# Using pandoc (if installed)
pandoc evaluation_results.tex -t html -o evaluation_results.html

# Or use the online converter at:
# https://pandoc.org/try/
```

## Troubleshooting

**Error: File not found**
- Make sure `evaluation_results.tex` is in the current directory

**Error: Undefined control sequence**
- Some packages may not be installed
- Install full TexLive: `sudo apt-get install texlive-full`

**The PDF looks different from expected**
- Run pdflatex twice (once for compilation, once for TOC)
- Some figures may not render without `pgfplots` package

## Quick Stats from Your Results

- **Evaluation Date:** December 23, 2025, 11:53:28
- **Queries Tested:** 5 MITRE ATT&CK questions
- **RAG Average:** 7.87/10
- **Graph+LLM Average:** 7.16/10
- **Winner:** RAG by 0.71 points
