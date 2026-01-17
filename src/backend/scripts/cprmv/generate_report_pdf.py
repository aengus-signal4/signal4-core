#!/usr/bin/env python3
"""
Generate PDF report for the CPRMV Misogyny Analysis.

Uses matplotlib for charts and FPDF for PDF generation.

Usage:
    cd ~/signal4/core
    python -m src.backend.scripts.cprmv.generate_report_pdf

    # Specify output
    python -m src.backend.scripts.cprmv.generate_report_pdf \
        --output ../frontend/public/data/cprmv-misogyny/report.pdf

    # Generate French version
    python -m src.backend.scripts.cprmv.generate_report_pdf \
        --output ../frontend/public/data/cprmv-misogyny/report-fr.pdf \
        --lang fr

Prerequisites:
    pip install fpdf2 matplotlib
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from fpdf import FPDF
except ImportError:
    print("Error: fpdf2 package required. Install with: pip install fpdf2")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('generate_pdf')

# Paths
AGGREGATIONS_PATH = Path(__file__).parent.parent.parent.parent.parent / "frontend/public/data/cprmv-misogyny/aggregations.json"
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent.parent.parent / "frontend/public/data/cprmv-misogyny/report.pdf"

# Colors
COLOR_EN = '#10b981'  # Emerald for English
COLOR_FR = '#d97706'  # Amber for French
COLOR_PRIMARY = '#64748b'  # Slate

# Translations
TRANSLATIONS = {
    'en': {
        'header': 'Misogyny in Canadian Audiovisual Content - Signal4 Research',
        'title_line1': 'Misogyny in Canadian',
        'title_line2': 'Audiovisual Content',
        'subtitle': 'Thematic analysis of misogynist discourse in Canadian far-right media',
        'organization': 'Signal4 Research',
        'executive_summary': 'Executive Summary',
        'executive_summary_text': (
            "This report demonstrates computational methods for detecting and analyzing "
            "misogynistic discourse in Canadian audiovisual content. Using advanced natural "
            "language processing, we analyzed content from {n_channels} channels, "
            "identifying {flagged_count:,} high-confidence instances of "
            "misogynistic language across {n_themes} thematic categories."
        ),
        'key_statistics': 'Key Statistics',
        'total_hours': 'Total Hours',
        'of_analyzed_content': 'of analyzed content',
        'episodes': 'Episodes',
        'podcast_episodes_videos': 'podcast episodes & videos',
        'flagged_segments': 'Flagged Segments',
        'containing_relevant_discourse': 'containing relevant discourse',
        'themes': 'Themes',
        'identified_thematic_categories': 'identified thematic categories',
        'date_range': 'Date Range',
        'corpus_overview': 'Corpus Overview',
        'corpus_overview_text': (
            "This dataset spans {total_episodes:,} videos/podcasts from "
            "{n_channels} distinct channels, totaling {total_hours:,.0f} hours "
            "of content produced between 2020 and 2025. The corpus represents a diverse ecosystem of "
            "Canadian far-right media, encompassing daily news broadcasts, long-form podcasts, "
            "livestream recordings, and short commentary videos."
        ),
        'content_production_by_language': 'Content Production by Language',
        'corpus_language_text': (
            "The corpus comprises {english_episodes:,} English episodes "
            "({english_hours:,.0f} hours) and {french_episodes:,} "
            "French episodes ({french_hours:,.0f} hours)."
        ),
        'flagged_content_analysis': 'Flagged Content Analysis',
        'flagged_analysis_text1': (
            "Of the {total_episodes:,} videos/podcasts in the corpus, "
            "{flagged_videos:,} ({flagged_pct:.1f}%) contain at least "
            "one segment classified as misogynistic discourse."
        ),
        'flagged_analysis_text2': (
            "Approximately {segment_pct:.2f}% of all segments in the corpus contain "
            "high-confidence misogynistic discourse (>=75% confidence). This represents "
            "{flagged_count:,} individual instances distributed across "
            "thousands of videos/podcasts."
        ),
        'thematic_analysis': 'Thematic Analysis',
        'theme_distribution': 'Theme Distribution',
        'theme_definitions': 'Theme Definitions',
        'channel_analysis': 'Channel Analysis',
        'top_channels_by_volume': 'Top Channels by Content Volume',
        'channels_by_rate': 'Channels by Flagged Content Rate',
        'channels_by_rate_text': (
            "The following chart shows channels ranked by the rate of flagged content "
            "(flagged segments per hour of content). Only channels with at least 10 hours of "
            "content are included."
        ),
        'methodology': 'Methodology',
        'processing_pipeline': 'Processing Pipeline',
        'processing_pipeline_text': (
            "The processing pipeline begins with content acquisition from Canadian channels spanning "
            "YouTube, podcasts, Rumble, and Odysee. Transcription employs automatic speech recognition "
            "(OpenAI Whisper and NVIDIA Parakeet). Speaker diarization uses pyannote to identify speaker "
            "boundaries. Each segment is embedded using a 2000-dimensional vector model (Qwen3-Embedding-4B)."
        ),
        'classification_pipeline': 'Classification Pipeline',
        'classification_pipeline_text': (
            "The classification pipeline employs a 7-stage approach designed to maximize precision "
            "while minimizing false positives:"
        ),
        'stages': [
            ("Stage 1 - Theme Definitions", "Thematic taxonomy with descriptions embedded for semantic matching."),
            ("Stage 2 - Semantic Search", "FAISS vector similarity search identifies candidate segments matching subtheme descriptions."),
            ("Stage 3 - Subtheme Classification", "The Qwen3-Next-80B-A3B model evaluates each candidate against specific subthemes."),
            ("Stage 4 - Validation", "Likert-scale evaluation (1-5) validates subtheme assignments, converting to confidence values."),
            ("Stage 5 - Relevance Verification", "High-confidence segments undergo verification to confirm relation to gender-based discourse."),
            ("Stage 6 - Speaker Stance Detection", "Identifies false positives by analyzing whether the speaker holds or rejects problematic views."),
            ("Stage 7 - Expanded Context Re-check", "False positives re-evaluated with ±20 seconds of surrounding context."),
        ],
        # Chart labels
        'chart_year': 'Year',
        'chart_hours_of_content': 'Hours of Content',
        'chart_content_by_lang_year': 'Content Production by Language and Year',
        'chart_english': 'English',
        'chart_french': 'French',
        'chart_num_segments': 'Number of Segments',
        'chart_theme_dist_top10': 'Theme Distribution (Top 10)',
        'chart_top_channels_volume': 'Top Channels by Content Volume',
        'chart_flagged_per_hour': 'Flagged Segments per Hour',
        'chart_channels_by_rate': 'Channels by Flagged Content Rate',
    },
    'fr': {
        'header': 'Misogynie dans le contenu audiovisuel canadien - Recherche Signal4',
        'title_line1': 'Misogynie dans le contenu',
        'title_line2': 'audiovisuel canadien',
        'subtitle': 'Analyse thématique du discours misogyne dans les médias canadiens d\'extrême droite',
        'organization': 'Recherche Signal4',
        'executive_summary': 'Résumé exécutif',
        'executive_summary_text': (
            "Ce rapport démontre des méthodes computationnelles pour détecter et analyser "
            "le discours misogyne dans le contenu audiovisuel canadien. En utilisant le traitement "
            "avancé du langage naturel, nous avons analysé le contenu de {n_channels} chaînes, "
            "identifiant {flagged_count:,} instances à haute confiance de "
            "langage misogyne réparties dans {n_themes} catégories thématiques."
        ),
        'key_statistics': 'Statistiques clés',
        'total_hours': 'Heures totales',
        'of_analyzed_content': 'de contenu analysé',
        'episodes': 'Épisodes',
        'podcast_episodes_videos': 'épisodes de podcast et vidéos',
        'flagged_segments': 'Segments signalés',
        'containing_relevant_discourse': 'contenant du discours pertinent',
        'themes': 'Thèmes',
        'identified_thematic_categories': 'catégories thématiques identifiées',
        'date_range': 'Période',
        'corpus_overview': 'Aperçu du corpus',
        'corpus_overview_text': (
            "Ce jeu de données couvre {total_episodes:,} vidéos/podcasts provenant de "
            "{n_channels} chaînes distinctes, totalisant {total_hours:,.0f} heures "
            "de contenu produit entre 2020 et 2025. Le corpus représente un écosystème diversifié de "
            "médias canadiens d'extrême droite, comprenant des bulletins d'information quotidiens, des podcasts "
            "de longue durée, des enregistrements de diffusions en direct et de courtes vidéos de commentaires."
        ),
        'content_production_by_language': 'Production de contenu par langue',
        'corpus_language_text': (
            "Le corpus comprend {english_episodes:,} épisodes en anglais "
            "({english_hours:,.0f} heures) et {french_episodes:,} "
            "épisodes en français ({french_hours:,.0f} heures)."
        ),
        'flagged_content_analysis': 'Analyse du contenu signalé',
        'flagged_analysis_text1': (
            "Des {total_episodes:,} vidéos/podcasts du corpus, "
            "{flagged_videos:,} ({flagged_pct:.1f}%) contiennent au moins "
            "un segment classé comme discours misogyne."
        ),
        'flagged_analysis_text2': (
            "Environ {segment_pct:.2f}% de tous les segments du corpus contiennent "
            "du discours misogyne à haute confiance (>=75% de confiance). Cela représente "
            "{flagged_count:,} instances individuelles réparties dans "
            "des milliers de vidéos/podcasts."
        ),
        'thematic_analysis': 'Analyse thématique',
        'theme_distribution': 'Distribution des thèmes',
        'theme_definitions': 'Définitions des thèmes',
        'channel_analysis': 'Analyse des chaînes',
        'top_channels_by_volume': 'Principales chaînes par volume de contenu',
        'channels_by_rate': 'Chaînes par taux de contenu signalé',
        'channels_by_rate_text': (
            "Le graphique suivant montre les chaînes classées par taux de contenu signalé "
            "(segments signalés par heure de contenu). Seules les chaînes avec au moins 10 heures de "
            "contenu sont incluses."
        ),
        'methodology': 'Méthodologie',
        'processing_pipeline': 'Pipeline de traitement',
        'processing_pipeline_text': (
            "Le pipeline de traitement commence par l'acquisition de contenu à partir de chaînes canadiennes "
            "sur YouTube, podcasts, Rumble et Odysee. La transcription utilise la reconnaissance vocale automatique "
            "(OpenAI Whisper et NVIDIA Parakeet). La diarisation des locuteurs utilise pyannote pour identifier les "
            "frontières des locuteurs. Chaque segment est encodé à l'aide d'un modèle vectoriel à 2000 dimensions (Qwen3-Embedding-4B)."
        ),
        'classification_pipeline': 'Pipeline de classification',
        'classification_pipeline_text': (
            "Le pipeline de classification utilise une approche en 7 étapes conçue pour maximiser la précision "
            "tout en minimisant les faux positifs :"
        ),
        'stages': [
            ("Étape 1 - Définitions des thèmes", "Taxonomie thématique avec descriptions encodées pour la correspondance sémantique."),
            ("Étape 2 - Recherche sémantique", "La recherche de similarité vectorielle FAISS identifie les segments candidats correspondant aux descriptions des sous-thèmes."),
            ("Étape 3 - Classification des sous-thèmes", "Le modèle Qwen3-Next-80B-A3B évalue chaque candidat par rapport à des sous-thèmes spécifiques."),
            ("Étape 4 - Validation", "L'évaluation sur échelle de Likert (1-5) valide les attributions de sous-thèmes, convertissant en valeurs de confiance."),
            ("Étape 5 - Vérification de pertinence", "Les segments à haute confiance sont vérifiés pour confirmer leur relation avec le discours basé sur le genre."),
            ("Étape 6 - Détection de la position du locuteur", "Identifie les faux positifs en analysant si le locuteur détient ou rejette des opinions problématiques."),
            ("Étape 7 - Revérification du contexte élargi", "Les faux positifs sont réévalués avec ±20 secondes de contexte environnant."),
        ],
        # Chart labels
        'chart_year': 'Année',
        'chart_hours_of_content': 'Heures de contenu',
        'chart_content_by_lang_year': 'Production de contenu par langue et année',
        'chart_english': 'Anglais',
        'chart_french': 'Français',
        'chart_num_segments': 'Nombre de segments',
        'chart_theme_dist_top10': 'Distribution des thèmes (Top 10)',
        'chart_top_channels_volume': 'Principales chaînes par volume de contenu',
        'chart_flagged_per_hour': 'Segments signalés par heure',
        'chart_channels_by_rate': 'Chaînes par taux de contenu signalé',
    }
}


class MisogynyReportPDF(FPDF):
    """Custom PDF class with header/footer."""

    def __init__(self, lang: str = 'en'):
        super().__init__()
        self.lang = lang
        self.t = TRANSLATIONS[lang]
        self.set_auto_page_break(auto=True, margin=20)
        # Add DejaVu fonts for Unicode support
        font_dir = Path.home() / 'Library/Fonts'
        self.add_font('DejaVu', style='', fname=str(font_dir / 'DejaVuSans.ttf'))
        self.add_font('DejaVu', style='B', fname=str(font_dir / 'DejaVuSans-Bold.ttf'))
        self.add_font('DejaVu', style='I', fname=str(font_dir / 'DejaVuSans-Oblique.ttf'))
        self.add_font('DejaVu', style='BI', fname=str(font_dir / 'DejaVuSans-BoldOblique.ttf'))

    def header(self):
        # No header - intentionally empty
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def chapter_title(self, title: str):
        self.set_font('DejaVu', 'B', 16)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def section_title(self, title: str):
        self.set_font('DejaVu', 'B', 13)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, ln=True)
        self.ln(3)

    def body_text(self, text: str):
        self.set_font('DejaVu', '', 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def stat_box(self, label: str, value: str, description: str = ""):
        self.set_font('DejaVu', 'B', 11)
        self.set_text_color(100, 100, 100)
        self.cell(45, 6, label + ":", ln=False)
        self.set_font('DejaVu', 'B', 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 6, value, ln=True)
        if description:
            self.set_font('DejaVu', 'I', 9)
            self.set_text_color(128, 128, 128)
            self.cell(45, 5, "", ln=False)
            self.cell(0, 5, description, ln=True)


def create_language_volume_chart(data: dict, t: dict) -> BytesIO:
    """Create stacked bar chart of content volume by language and year."""
    lang_data = data['lang_volume_by_year']

    # Group by year
    years = sorted(set(d['year'] for d in lang_data))
    en_hours = []
    fr_hours = []

    for year in years:
        en = sum(d['hours'] for d in lang_data if d['year'] == year and d['main_language'] == 'en')
        fr = sum(d['hours'] for d in lang_data if d['year'] == year and d['main_language'] == 'fr')
        en_hours.append(en)
        fr_hours.append(fr)

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(years))
    width = 0.7

    ax.bar(x, en_hours, width, label=t['chart_english'], color=COLOR_EN, alpha=0.85)
    ax.bar(x, fr_hours, width, bottom=en_hours, label=t['chart_french'], color=COLOR_FR, alpha=0.85)

    ax.set_xlabel(t['chart_year'], fontsize=10)
    ax.set_ylabel(t['chart_hours_of_content'], fontsize=10)
    ax.set_title(t['chart_content_by_lang_year'], fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def create_theme_distribution_chart(data: dict, t: dict, lang: str = 'en') -> BytesIO:
    """Create horizontal bar chart of theme distribution."""
    themes = data['theme_distribution'][:10]  # Top 10

    # Use French theme names if available and lang is French
    if lang == 'fr':
        names = [th.get('theme_name_fr', th['theme_name'])[:40] for th in themes]
    else:
        names = [th['theme_name'][:40] for th in themes]  # Truncate long names
    counts = [th['segment_count'] for th in themes]

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(names))
    ax.barh(y, counts, color=COLOR_PRIMARY, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(t['chart_num_segments'], fontsize=10)
    ax.set_title(t['chart_theme_dist_top10'], fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Add count labels
    for i, v in enumerate(counts):
        ax.text(v + max(counts)*0.01, i, f'{v:,}', va='center', fontsize=9)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def create_channel_volume_chart(data: dict, t: dict) -> BytesIO:
    """Create horizontal bar chart of top channels by volume."""
    channels = data['circle_pack_by_volume'][:15]  # Top 15

    names = [c['channel_name'][:30] for c in channels]
    hours = [c['total_hours'] for c in channels]
    colors = [COLOR_EN if c['main_language'] == 'en' else COLOR_FR for c in channels]

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(names))
    ax.barh(y, hours, color=colors, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(t['chart_hours_of_content'], fontsize=10)
    ax.set_title(t['chart_top_channels_volume'], fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Legend
    en_patch = mpatches.Patch(color=COLOR_EN, label=t['chart_english'])
    fr_patch = mpatches.Patch(color=COLOR_FR, label=t['chart_french'])
    ax.legend(handles=[en_patch, fr_patch], loc='lower right')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def create_channel_rate_chart(data: dict, t: dict) -> BytesIO:
    """Create horizontal bar chart of channels by flagged segments per hour."""
    channels = data['circle_pack_by_rate'][:15]  # Top 15

    names = [c['channel_name'][:30] for c in channels]
    rates = [c['segments_per_hour'] for c in channels]
    colors = [COLOR_EN if c['main_language'] == 'en' else COLOR_FR for c in channels]

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(names))
    ax.barh(y, rates, color=colors, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(t['chart_flagged_per_hour'], fontsize=10)
    ax.set_title(t['chart_channels_by_rate'], fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Legend
    en_patch = mpatches.Patch(color=COLOR_EN, label=t['chart_english'])
    fr_patch = mpatches.Patch(color=COLOR_FR, label=t['chart_french'])
    ax.legend(handles=[en_patch, fr_patch], loc='lower right')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def generate_pdf(data: dict, output_path: Path, lang: str = 'en'):
    """Generate the full PDF report."""
    pdf = MisogynyReportPDF(lang=lang)
    pdf.alias_nb_pages()
    t = pdf.t

    stats = data.get('summary_stats', {})

    # Format date based on language
    if lang == 'fr':
        # French month names
        months_fr = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin',
                     'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
        now = datetime.now()
        date_str = f"{months_fr[now.month - 1]} {now.year}"
    else:
        date_str = datetime.now().strftime('%B %Y')

    # === Title Page ===
    pdf.add_page()
    pdf.ln(40)

    # Title (with line break between lines)
    pdf.set_font('DejaVu', 'B', 28)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, t['title_line1'], ln=True, align='C')
    pdf.cell(0, 15, t['title_line2'], ln=True, align='C')
    pdf.ln(10)
    pdf.set_font('DejaVu', '', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, t['subtitle'], ln=True, align='C')

    pdf.ln(30)

    # Prepared for and date
    pdf.set_font('DejaVu', '', 12)
    pdf.set_text_color(60, 60, 60)
    prepared_for = 'Préparé pour le CPRMV' if lang == 'fr' else 'Prepared for CPRMV'
    pdf.cell(0, 8, prepared_for, ln=True, align='C')
    pdf.cell(0, 8, date_str, ln=True, align='C')

    # === Executive Summary ===
    pdf.add_page()
    pdf.chapter_title(t['executive_summary'])

    pdf.body_text(
        t['executive_summary_text'].format(
            n_channels=stats.get('n_channels', 'N/A'),
            flagged_count=data.get('flagged_segment_count', 0),
            n_themes=len(data.get('themes', []))
        )
    )

    pdf.ln(5)
    pdf.section_title(t['key_statistics'])
    pdf.stat_box(t['total_hours'], f"{data.get('total_hours', 0):,.0f}", t['of_analyzed_content'])
    pdf.stat_box(t['episodes'], f"{data.get('total_episodes', 0):,}", t['podcast_episodes_videos'])
    pdf.stat_box(t['flagged_segments'], f"{data.get('flagged_segment_count', 0):,}", t['containing_relevant_discourse'])
    pdf.stat_box(t['themes'], f"{len(data.get('themes', []))}", t['identified_thematic_categories'])
    pdf.stat_box(t['date_range'], f"{stats.get('min_date', '')} - {stats.get('max_date', '')}")

    # === Corpus Overview ===
    pdf.add_page()
    pdf.chapter_title(t['corpus_overview'])

    pdf.body_text(
        t['corpus_overview_text'].format(
            total_episodes=data.get('total_episodes', 0),
            n_channels=stats.get('n_channels', 0),
            total_hours=data.get('total_hours', 0)
        )
    )

    pdf.ln(5)
    pdf.section_title(t['content_production_by_language'])

    # Language volume chart
    chart_buf = create_language_volume_chart(data, t)
    pdf.image(chart_buf, x=10, w=190)

    pdf.ln(5)
    pdf.body_text(
        t['corpus_language_text'].format(
            english_episodes=stats.get('english_episodes', 0),
            english_hours=stats.get('english_hours', 0),
            french_episodes=stats.get('french_episodes', 0),
            french_hours=stats.get('french_hours', 0)
        )
    )

    # === Flagged Content Analysis ===
    pdf.add_page()
    pdf.chapter_title(t['flagged_content_analysis'])

    pdf.body_text(
        t['flagged_analysis_text1'].format(
            total_episodes=data.get('total_episodes', 0),
            flagged_videos=stats.get('flagged_videos', 0),
            flagged_pct=stats.get('flagged_pct', 0)
        )
    )

    pdf.body_text(
        t['flagged_analysis_text2'].format(
            segment_pct=stats.get('segment_pct', 0),
            flagged_count=data.get('flagged_segment_count', 0)
        )
    )

    # === Themes ===
    pdf.add_page()
    pdf.chapter_title(t['thematic_analysis'])

    pdf.section_title(t['theme_distribution'])
    chart_buf = create_theme_distribution_chart(data, t, lang)
    pdf.image(chart_buf, x=10, w=190)

    pdf.ln(5)
    pdf.section_title(t['theme_definitions'])
    for theme in data.get('themes', [])[:10]:
        pdf.set_font('DejaVu', 'B', 10)
        pdf.set_text_color(0, 0, 0)
        # Use French theme name if available
        theme_name = theme.get('theme_name_fr', theme.get('theme_name', 'Unknown')) if lang == 'fr' else theme.get('theme_name', 'Unknown')
        pdf.cell(0, 6, f"• {theme_name}", ln=True)
        # Use French description if available
        desc_key = 'description_fr' if lang == 'fr' else 'description_en'
        desc = theme.get(desc_key) or theme.get('description_en', '')
        if desc:
            pdf.set_font('DejaVu', '', 9)
            pdf.set_text_color(80, 80, 80)
            desc = desc[:200] + '...' if len(desc) > 200 else desc
            pdf.multi_cell(0, 4, f"  {desc}")
        pdf.ln(2)

    # === Channels ===
    pdf.add_page()
    pdf.chapter_title(t['channel_analysis'])

    pdf.section_title(t['top_channels_by_volume'])
    chart_buf = create_channel_volume_chart(data, t)
    pdf.image(chart_buf, x=10, w=190)

    pdf.add_page()
    pdf.section_title(t['channels_by_rate'])
    pdf.body_text(t['channels_by_rate_text'])
    chart_buf = create_channel_rate_chart(data, t)
    pdf.image(chart_buf, x=10, w=190)

    # === Methodology ===
    pdf.add_page()
    pdf.chapter_title(t['methodology'])

    pdf.section_title(t['processing_pipeline'])
    pdf.body_text(t['processing_pipeline_text'])

    pdf.section_title(t['classification_pipeline'])
    pdf.body_text(t['classification_pipeline_text'])

    for stage, desc in t['stages']:
        pdf.set_font('DejaVu', 'B', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, stage, ln=True)
        pdf.set_font('DejaVu', '', 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4, desc)
        pdf.ln(2)

    # Save
    pdf.output(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate CPRMV Misogyny Report PDF')
    parser.add_argument(
        '--output',
        type=str,
        default=str(DEFAULT_OUTPUT),
        help='Output PDF path'
    )
    parser.add_argument(
        '--aggregations',
        type=str,
        default=str(AGGREGATIONS_PATH),
        help='Path to aggregations.json'
    )
    parser.add_argument(
        '--lang',
        type=str,
        choices=['en', 'fr'],
        default='en',
        help='Language for report (en or fr)'
    )

    args = parser.parse_args()
    output_path = Path(args.output)
    aggregations_path = Path(args.aggregations)
    lang = args.lang

    lang_name = 'French' if lang == 'fr' else 'English'
    logger.info("=" * 60)
    logger.info(f"Generating CPRMV Misogyny Report PDF ({lang_name})")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading data from {aggregations_path}...")
    with open(aggregations_path) as f:
        data = json.load(f)

    logger.info(f"  Themes: {len(data.get('themes', []))}")
    logger.info(f"  Flagged segments: {data.get('flagged_segment_count', 0):,}")
    logger.info(f"  Language: {lang_name}")

    # Generate PDF
    logger.info(f"Generating PDF...")
    generate_pdf(data, output_path, lang=lang)

    file_size = output_path.stat().st_size / 1024
    logger.info("")
    logger.info("=" * 60)
    logger.info("PDF Generated Successfully")
    logger.info("=" * 60)
    logger.info(f"Output: {output_path}")
    logger.info(f"Size: {file_size:.1f} KB")

    return 0


if __name__ == '__main__':
    sys.exit(main())
