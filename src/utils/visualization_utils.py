import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import textwrap
import json
from pathlib import Path
import logging
from typing import List, Dict, Set, Optional
import traceback
import sys

logger = logging.getLogger(__name__)

# Ensure logger is properly configured
def setup_viz_logger():
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

def visualize_speaker_turns(
    content_id: str,
    final_turns: List[Dict],
    mapped_dia_segments: List[Dict],
    output_dir: Path,
    found_global_ids: Set[str]
) -> Optional[Path]:
    """
    Visualize speaker turns and diarization segments.
    
    Args:
        content_id: The content ID being processed
        final_turns: List of final speaker turns
        mapped_dia_segments: List of mapped diarization segments
        output_dir: Directory to save the visualization
        found_global_ids: Set of global speaker IDs found
        
    Returns:
        Path to the saved visualization file if successful, None otherwise
    """
    setup_viz_logger()
    
    try:
        logger.info(f"Starting visualization for content_id: {content_id}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Number of final turns: {len(final_turns) if final_turns else 0}")
        logger.info(f"Number of dia segments: {len(mapped_dia_segments) if mapped_dia_segments else 0}")
        logger.info(f"Found global IDs: {found_global_ids}")
        
        if not final_turns:
            logger.warning("No final turns to visualize")
            return None

        logger.debug("Creating output directory if it doesn't exist")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine unique speakers and assign Y-positions
        speakers = sorted(list(found_global_ids)) + ["Unassigned"]
        speaker_y_pos = {speaker: i for i, speaker in enumerate(speakers)}
        num_speakers = len(speakers)
        logger.debug(f"Identified {num_speakers} unique speakers")

        # Generate distinct colors
        logger.debug("Setting up color mapping")
        colors = plt.cm.get_cmap('tab20', num_speakers)
        speaker_colors = {speaker: colors(i) for i, speaker in enumerate(speakers)}
        speaker_colors["Unassigned"] = 'grey'

        # Calculate figure dimensions
        logger.debug("Calculating figure dimensions")
        max_time_turn = max(t['end'] for t in final_turns) if final_turns else 0
        max_time_dia = max(d['end'] for d in mapped_dia_segments) if mapped_dia_segments else 0
        max_time = max(max_time_turn, max_time_dia)
        logger.debug(f"Max time: {max_time} seconds")

        inches_per_second = 1 / 5
        dynamic_width = max(20, max_time * inches_per_second)
        figure_height = num_speakers * 0.6 + 2
        logger.debug(f"Figure dimensions: {dynamic_width}x{figure_height} inches")

        logger.debug("Creating matplotlib figure")
        fig, ax = plt.subplots(figsize=(dynamic_width, figure_height))

        # Plot diarization segments
        logger.debug("Plotting diarization segments")
        for dia_seg in mapped_dia_segments:
            speaker = dia_seg['speaker']
            if speaker not in speaker_y_pos:
                logger.warning(f"Diarization speaker '{speaker}' not in speaker_y_pos. Skipping.")
                continue
                
            y = speaker_y_pos[speaker]
            start = dia_seg['start']
            duration = dia_seg['end'] - start
            ax.barh(y, duration, left=start, height=0.5,
                    color=speaker_colors[speaker], alpha=0.4, edgecolor='black',
                    label=f"Dia: {speaker}" if f"Dia: {speaker}" not in ax.get_legend_handles_labels()[1] else "")

        # Plot text blocks with optimized wrapping
        logger.debug("Plotting speaker turns with text")
        text_fontsize = 7
        chars_per_second = 6
        min_wrap_width = 12

        for turn in final_turns:
            speaker = turn.get('speaker', "Unassigned")
            if speaker not in speaker_y_pos:
                logger.warning(f"Speaker '{speaker}' not in found_global_ids. Assigning to Unassigned.")
                speaker = "Unassigned"

            y = speaker_y_pos[speaker]
            turn_start = turn['start']
            turn_end = turn['end']
            turn_duration = turn_end - turn_start
            turn_mid_time = turn_start + (turn_end - turn_start) / 2
            color = speaker_colors[speaker]

            # Calculate optimal wrap width based on duration
            wrap_width = max(min_wrap_width, int(turn_duration * chars_per_second))
            
            # Optimize text wrapping
            text = turn['text']
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= wrap_width:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Ensure minimum and maximum lines
            if len(lines) < 2:
                lines = textwrap.wrap(text, wrap_width, break_long_words=True)
            elif len(lines) > 10:
                lines = textwrap.wrap(text, wrap_width * 2, break_long_words=True)
            
            wrapped_text = '\n'.join(lines)

            # Position text above or below center line
            vertical_offset = 0.2 if (speaker_y_pos[speaker] % 2 == 0) else -0.2
            vertical_alignment = 'bottom' if vertical_offset > 0 else 'top'

            # Add text with box
            bbox_props = dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec=color, lw=0.5)
            ax.text(turn_mid_time, y + vertical_offset, wrapped_text,
                    ha='center', va=vertical_alignment, fontsize=text_fontsize, color=color,
                    bbox=bbox_props)
            
            # Add time span line
            ax.plot([turn_start, turn_end], [y, y], color=color, linewidth=2, solid_capstyle='butt')

        # Configure plot
        logger.debug("Configuring plot layout")
        ax.set_yticks(range(num_speakers))
        ax.set_yticklabels(speakers)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Speaker")
        ax.set_title(f"Word-Speaker Assignment Visualization for {content_id}")
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.05, 1.0))

        # Set axis limits
        ax.set_xlim(0, max_time * 1.01)
        ax.set_ylim(-0.5, num_speakers - 0.5)

        # Save plot
        logger.debug(f"Saving plot to: {output_dir}")
        plot_filename = output_dir / f"{content_id}_word_speaker_assignment.png"
        logger.info(f"Saving visualization to: {plot_filename}")
        plt.savefig(plot_filename, dpi=90)
        plt.close(fig)

        logger.info(f"Visualization saved successfully to: {plot_filename}")
        return plot_filename

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None 