#!/usr/bin/env Rscript
# =============================================================================
# American Reactions to Canada at Davos - Circle Pack Visualization
# =============================================================================

library(tidyverse)
library(jsonlite)
library(packcircles)
library(ggplot2)
library(ggtext)

# MEO Theme colors
MEO_ORANGE <- "#FF8200"
MEO_GREEN <- "#467742"
MEO_BURGUNDY <- "#6D4A4D"
MEO_BLUE <- "#434E7C"
MEO_TEAL <- "#6BADC6"
MEO_DARK <- "#272B26"
MEO_GRAY <- "#D7D6D4"

# Load MEO theme
source("~/aide/visualization_guides/meo/meo_theme.R")

# =============================================================================
# Load Data
# =============================================================================

cache_dir <- "core/meo/us-can/.cache"
output_dir <- "core/meo/us-can/outputs"

segments <- fromJSON(file.path(cache_dir, "davos_reaction_segments.json"))
classifications <- fromJSON(file.path(cache_dir, "davos_classifications.json"))

# Create lookup
segment_lookup <- segments %>%
  select(segment_id, text, channel_name) %>%
  distinct()

# Join classifications with segments
classified <- classifications %>%
  filter(is_relevant == TRUE) %>%
  left_join(segment_lookup, by = "segment_id")

cat("Loaded", nrow(classified), "relevant classifications\n")

# =============================================================================
# Category Content (from sub-agent analysis)
# =============================================================================

# 2x2 grid content: [row=trump stance, col=canada stance]
# Top row = Pro-Trump, Bottom row = Anti-Trump
# Left col = Anti-Canada, Right col = Pro-Canada
category_content <- list(
  "Pro-Canada\nPro-Trump" = list(
    label = "Allied strength",
    synthesis = "Trump's NATO pressure strengthens Western alliance. Arctic defense benefits both nations.",
    quote1 = "Carney, as critical as he was...I think the president respects that. He doesn't do well if the other side are weaklings.",
    source1 = "The Five",
    quote2 = "Canada came to 2%. Without Donald Trump, this would never have happened.",
    source2 = "Wendy Bell Radio"
  ),
  "Pro-Canada\nAnti-Trump" = list(
    label = "Dignified resistance",
    synthesis = "Carney portrayed as principled statesman. His 'rupture' warning seen as truth-telling about American unreliability.",
    quote1 = "Ashamed of what we saw at Davos...jealous of the integrity and intellect of Prime Minister Carney.",
    source1 = "Sara Gonzales Unfiltered",
    quote2 = "Contrast it with Carney's speech, it's two different worlds...you have a delusional Donald.",
    source2 = "The 11th Hour"
  ),
  "Anti-Canada\nPro-Trump" = list(
    label = "Ungrateful ally",
    synthesis = "Canada framed as weak, ungrateful ally freeloading off American security, now treacherously pivoting to China.",
    quote1 = "Canada lives because of the United States. Remember that, Mark.",
    source1 = "Markley, van Camp & Robbins",
    quote2 = "Canada...we can't work with America, so we're going to make a deal with China...this is a stupid move. You should not do this.",
    source2 = "Clay Travis and Buck Sexton"
  ),
  "Anti-Canada\nAnti-Trump" = list(
    label = "Broken system",
    synthesis = "Both dismissed as culpable in failing international order. Carney as elitist; Trump as chaotic.",
    quote1 = "Carney's speech struck me as just as petulant as the president's initiative.",
    source1 = "The Bulletin",
    quote2 = "Mark Carney is an embarrassment...Trump will be gone from the scene.",
    source2 = "Armstrong & Getty"
  )
)

# =============================================================================
# Prepare Data - All 4 quadrants
# =============================================================================

categories <- classified %>%
  mutate(
    category = case_when(
      stance_canada == "SUPPORTIVE" & stance_trump == "SUPPORTIVE" ~ "Pro-Canada\nPro-Trump",
      stance_canada == "SUPPORTIVE" & stance_trump == "CRITICAL" ~ "Pro-Canada\nAnti-Trump",
      stance_canada == "CRITICAL" & stance_trump == "SUPPORTIVE" ~ "Anti-Canada\nPro-Trump",
      stance_canada == "CRITICAL" & stance_trump == "CRITICAL" ~ "Anti-Canada\nAnti-Trump",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(category))

category_counts <- categories %>%
  group_by(category) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count))

cat("\nCategory counts:\n")
print(category_counts)

# =============================================================================
# Create 2x2 Grid Layout - Filled Squares
# =============================================================================

# Add content from category_content
grid_data <- category_counts %>%
  rowwise() %>%
  mutate(
    label = category_content[[category]]$label,
    synthesis = category_content[[category]]$synthesis,
    quote1 = category_content[[category]]$quote1,
    source1 = category_content[[category]]$source1,
    quote2 = category_content[[category]]$quote2,
    source2 = category_content[[category]]$source2
  ) %>%
  ungroup()

# Grid positions: columns = Canada stance, rows = Trump stance
# Each cell is 1x1, grid spans from -1 to 1 on both axes
grid_data <- grid_data %>%
  mutate(
    # Opacity based on count (more segments = more opaque)
    alpha = count / max(count),
    # Cell positions (center of each cell)
    x = case_when(
      grepl("Anti-Canada", category) ~ -0.5,
      grepl("Pro-Canada", category) ~ 0.5
    ),
    y = case_when(
      grepl("Pro-Trump", category) ~ 0.5,
      grepl("Anti-Trump", category) ~ -0.5
    ),
    # Cell boundaries
    xmin = x - 0.5,
    xmax = x + 0.5,
    ymin = y - 0.5,
    ymax = y + 0.5
  )

# =============================================================================
# Build Visualization
# =============================================================================

p <- ggplot() +
  # Draw filled squares with black borders, orange fill with varying opacity
  geom_rect(
    data = grid_data,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, alpha = alpha),
    fill = MEO_ORANGE,
    color = MEO_DARK,
    linewidth = 1.5
  ) +
  scale_alpha_identity() +

  # Category labels inside cells
  geom_text(
    data = grid_data,
    aes(x = x, y = y + 0.22, label = label),
    size = 5,
    fontface = "bold",
    color = MEO_DARK,
    lineheight = 0.9
  ) +

  # Segment counts
  geom_text(
    data = grid_data,
    aes(x = x, y = y + 0.05, label = paste0(count, " segments")),
    size = 4,
    color = MEO_DARK,
    fontface = "bold"
  ) +

  # Synthesis text inside cells
  geom_text(
    data = grid_data,
    aes(x = x, y = y - 0.12, label = str_wrap(synthesis, width = 35)),
    size = 2.8,
    color = MEO_DARK,
    lineheight = 1.0,
    vjust = 1
  ) +

  # Axis labels - Canada (above each column)
  annotate("text", x = -0.5, y = 1.08,
           label = "Anti-Canada", size = 5, fontface = "bold", color = MEO_DARK) +
  annotate("text", x = 0.5, y = 1.08,
           label = "Pro-Canada", size = 5, fontface = "bold", color = MEO_DARK) +

  # Axis labels - Trump (left of each row, rotated) - close to grid
  annotate("text", x = -1.08, y = 0.5,
           label = "Pro-Trump", size = 5, fontface = "bold", color = MEO_DARK, angle = 90) +
  annotate("text", x = -1.08, y = -0.5,
           label = "Anti-Trump", size = 5, fontface = "bold", color = MEO_DARK, angle = 90) +

  coord_equal(clip = "off") +
  theme_void() +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5, color = MEO_DARK, margin = margin(b = 20)),
    plot.caption = element_text(size = 8, hjust = 1, color = MEO_DARK, margin = margin(t = 30), lineheight = 1.2),
    plot.caption.position = "plot",
    plot.margin = margin(40, 60, 30, 100)
  ) +

  labs(
    title = "US News/Politics/Culture podcasts on Canada after Davos speech",
    caption = str_wrap(paste0(
      "Based on ", nrow(classified), " segments from ~300 top US News/Politics/Culture podcasts (Jan 21-23, 2026). ",
      "Segments mentioning Canada/Carney classified on stance toward Canada and Trump. ",
      "Centre for Media, Technology, and Democracy"
    ), width = 90)
  )

# Add quotes - each cell gets one quote ABOVE/BELOW and one quote LEFT/RIGHT
# Format: "quote" -Channel Name

# Helper to format quote with source on same line
fmt_quote <- function(quote, source, width) {
  str_wrap(paste0('"', quote, '" -', source), width = width)
}

# Helper for top/bottom quotes (tighter wrap)
fmt_quote_tb <- function(quote, source, width) {
  str_wrap(paste0('"', quote, '" -', source), width = width)
}

# Anti-Canada/Pro-Trump (top-left cell)
acpt_row <- grid_data %>% filter(category == "Anti-Canada\nPro-Trump")
if (nrow(acpt_row) > 0) {
  p <- p +
    # Quote above
    annotate("text", x = -0.5, y = 1.18,
             label = fmt_quote(acpt_row$quote1[1], acpt_row$source1[1], 30),
             size = 2.8, hjust = 0.5, vjust = 0, color = MEO_DARK, lineheight = 1.0) +
    # Quote left
    annotate("text", x = -1.18, y = 0.5,
             label = fmt_quote(acpt_row$quote2[1], acpt_row$source2[1], 28),
             size = 2.8, hjust = 1, vjust = 0.5, color = MEO_DARK, lineheight = 1.0)
}

# Pro-Canada/Pro-Trump (top-right cell)
pcpt_row <- grid_data %>% filter(category == "Pro-Canada\nPro-Trump")
if (nrow(pcpt_row) > 0) {
  p <- p +
    # Quote above
    annotate("text", x = 0.5, y = 1.18,
             label = fmt_quote(pcpt_row$quote1[1], pcpt_row$source1[1], 38),
             size = 2.8, hjust = 0.5, vjust = 0, color = MEO_DARK, lineheight = 1.0) +
    # Quote right
    annotate("text", x = 1.05, y = 0.5,
             label = fmt_quote(pcpt_row$quote2[1], pcpt_row$source2[1], 28),
             size = 2.8, hjust = 0, vjust = 0.5, color = MEO_DARK, lineheight = 1.0)
}

# Anti-Canada/Anti-Trump (bottom-left cell)
acat_row <- grid_data %>% filter(category == "Anti-Canada\nAnti-Trump")
if (nrow(acat_row) > 0) {
  p <- p +
    # Quote below
    annotate("text", x = -0.5, y = -1.08,
             label = fmt_quote(acat_row$quote1[1], acat_row$source1[1], 38),
             size = 2.8, hjust = 0.5, vjust = 1, color = MEO_DARK, lineheight = 1.0) +
    # Quote left
    annotate("text", x = -1.18, y = -0.5,
             label = fmt_quote(acat_row$quote2[1], acat_row$source2[1], 28),
             size = 2.8, hjust = 1, vjust = 0.5, color = MEO_DARK, lineheight = 1.0)
}

# Pro-Canada/Anti-Trump (bottom-right cell)
pcat_row <- grid_data %>% filter(category == "Pro-Canada\nAnti-Trump")
if (nrow(pcat_row) > 0) {
  p <- p +
    # Quote below
    annotate("text", x = 0.5, y = -1.08,
             label = fmt_quote(pcat_row$quote1[1], pcat_row$source1[1], 38),
             size = 2.8, hjust = 0.5, vjust = 1, color = MEO_DARK, lineheight = 1.0) +
    # Quote right
    annotate("text", x = 1.05, y = -0.5,
             label = fmt_quote(pcat_row$quote2[1], pcat_row$source2[1], 28),
             size = 2.8, hjust = 0, vjust = 0.5, color = MEO_DARK, lineheight = 1.0)
}

ggsave(
  file.path(output_dir, "04_discourse.png"),
  p,
  width = 10,
  height = 8,
  dpi = 300,
  bg = "white"
)

cat("\nSaved:", file.path(output_dir, "04_discourse.png"), "\n")
cat("Done!\n")
