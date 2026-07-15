# NMA-inspired Beamer template

This directory contains a 16:9 Beamer theme based on the recurring visual
language of Neuromatch Academy course slides: a charcoal title/section frame,
white content frames, a flush-bottom metadata bar, and restrained teal, purple,
and coral accents.

## Build the demo

```sh
cd presentation
make
```

The default build uses LuaLaTeX and prefers the Homebrew-distributed Metropolis
family, a close fit for the geometric NMA course-slide typography. It falls
back to Proxima Nova and then TeX Gyre Heros; pdfLaTeX is also supported with
Latin Modern Sans. The Makefile exposes the standard macOS user and system font
directories through `OSFONTDIR`, so LuaLaTeX can resolve Homebrew cask fonts.
Equations use STIX Two Math under LuaLaTeX/XeLaTeX, with the matching STIX2
Type 1 math package under pdfLaTeX.

## Use the theme

Keep `beamerthemeNMA.sty` beside your presentation and start with:

```tex
\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{NMA}

\title[Short title]{Presentation title}
\author[Short name]{Presenter name}
\course{Computational Neuroscience}
\week{Week 1}
\session{Day 1}
\topic{Intro}
```

The theme adds two full-bleed helpers:

```tex
\NMAsectionframe[Optional subtitle]{Section title}
\NMAclosingframe[Optional subtitle]{Closing statement}
```

Use `\NMAtakeaway{...}` for one concise highlighted conclusion. The theme uses
the official Neuromatch Academy SVG wordmark and logo mark from the maintained
[`neuromatch/neuromatch.io`](https://github.com/neuromatch/neuromatch.io/tree/main/static/svgs/logos)
repository. Matching PDF conversions are included for direct, vector-only
Beamer output. Override `\NMAassetpath` if the `assets/` directory is not next
to the theme file.

For plots and diagrams, prefer PDF or TikZ; reserve PNG/JPEG for inherently
raster material such as photographs and microscopy.

See `demo.tex` for title, section, block, equation, two-column, and closing
frame examples.
