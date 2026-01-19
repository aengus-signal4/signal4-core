#!/usr/bin/env python3
"""
CPRMV Pipeline Animation - Narrative journey through the classification pipeline.
Run with: manim -pqh pipeline_animation.py CPRMVAnimation
"""

from manim import *
import numpy as np

# Color scheme
CHANNEL_BLUE = "#4A90D9"
EPISODE_BLUE = "#6495ED"
SEGMENT_GRAY = "#888888"
CANDIDATE_GREEN = "#50C878"
INCLUDED_GREEN = "#228B22"
EXCLUDED_RED = "#CD5C5C"
ORANGE = "#FFA500"
BARRIER_GLOW = "#00FFFF"


class CPRMVAnimation(Scene):
    def layout_block(self, mobjects, cols, center, spacing=0.16):
        """Return positions for mobjects arranged in a tight grid block."""
        positions = []
        rows = int(np.ceil(len(mobjects) / cols)) if cols else 0
        for i, _ in enumerate(mobjects):
            r = i // cols
            c = i % cols
            x = center[0] + (c - (cols - 1) / 2) * spacing
            y = center[1] - (r - (rows - 1) / 2) * spacing
            positions.append(np.array([x, y, center[2]]))
        return positions
    def construct(self):
        # Pipeline statistics
        self.channels = 45
        self.episodes = 48724
        self.total_segments = 1141825
        self.stage1_candidates = 231905
        self.stage3_excluded = 184512
        self.stage3_passed = 47393
        self.stage4_holds = 41539
        self.stage4_rejects = 5854
        self.stage5_recovered = 1862
        self.stage5_fp = 3992
        self.final_included = 43401

        # Run the animation parts
        self.part1_channels()
        self.part2_episodes()
        # Cut after stage 2 (episodes)
        # self.part3_segmentation()
        # self.part4_classification_gates()
        # self.part5_final_result()

    def part1_channels(self):
        """Part 1: Show podcast channel sources as circular icons flying in."""
        import os
        from PIL import Image, ImageDraw, ImageFilter

        title_main = Text("53 Content", font_size=28, color=WHITE)
        title_sub = Text("Channels", font_size=28, color=WHITE)
        title_group = VGroup(title_main, title_sub).arrange(RIGHT, buff=0.15)
        title_group.to_edge(UP, buff=0.4)

        # Get channel icon files - use absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_dir = os.path.join(script_dir, "channel_icons")
        icon_files = sorted([f for f in os.listdir(icon_dir) if f.endswith('.jpg')])

        # Create 5x8 grid (40 icons) + 13 placeholder circles = 53 channels
        channels = VGroup()  # Circle borders
        channel_images = Group()  # Images
        n_cols, n_rows = 8, 5
        radius = 0.38
        spacing_x = 0.95
        spacing_y = 0.95

        idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                # Final grid position
                x = (col - (n_cols - 1) / 2) * spacing_x
                y = ((n_rows - 1) / 2 - row) * spacing_y - 0.3

                # Start position (off screen left)
                start_x = -9 - (row * 0.5) - (col * 0.2)

                # Create circle border
                circle = Circle(
                    radius=radius,
                    color=CHANNEL_BLUE,
                    fill_opacity=0,
                    stroke_width=0,  # hide borders
                )
                circle.move_to([start_x, y, 0])

                # Add image if available (first 40 have icons)
                if idx < len(icon_files):
                    icon_path = os.path.join(icon_dir, icon_files[idx])

                    # Load and process image: make circular and grayscale
                    pil_img = Image.open(icon_path).convert('L')  # Grayscale
                    pil_img = pil_img.resize((200, 200), Image.LANCZOS)

                    # Create circular mask
                    mask = Image.new('L', (200, 200), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((0, 0, 200, 200), fill=255)

                    # Apply mask
                    circular_img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
                    gray_rgba = Image.merge('RGBA', (pil_img, pil_img, pil_img, mask))
                    circular_img.paste(gray_rgba, (0, 0))

                    # Save temp file and load as ImageMobject
                    temp_path = f"/tmp/channel_icon_{idx}.png"
                    circular_img.save(temp_path)

                    img = ImageMobject(temp_path)
                    img.height = radius * 2
                    img.width = radius * 2
                    img.move_to([start_x, y, 0])
                    channel_images.add(img)
                else:
                    # Placeholder for channels without icons (make filled circle)
                    circle.set_fill(GRAY, opacity=0.3)

                channels.add(circle)
                idx += 1

        # Animate title
        self.play(Write(title_group), run_time=0.8)

        # Fly in from left - images and circles together
        final_positions = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = (col - (n_cols - 1) / 2) * spacing_x
                y = ((n_rows - 1) / 2 - row) * spacing_y - 0.3
                final_positions.append([x, y, 0])

        # Animate flying in with stagger
        fly_anims = []
        for i, (circle, pos) in enumerate(zip(channels, final_positions)):
            fly_anims.append(circle.animate.move_to(pos))
        for i, img in enumerate(channel_images):
            if i < len(final_positions):
                fly_anims.append(img.animate.move_to(final_positions[i]))

        self.play(
            LaggedStart(*fly_anims, lag_ratio=0.02),
            run_time=2.5
        )

        self.wait(0.5)

        # Store references
        self.channels_group = channels
        self.channel_images = channel_images
        self.title_main = title_main
        self.title_sub = title_sub
        self.title_group = title_group

        # Move channels to compact column on left (transition to part 2)
        # Grid will be 10 rows x 4 cols (for 40 icons), centered at x=-5.5
        grid_center_x = -5.5
        grid_top_y = 2.0
        grid_spacing = 0.36
        n_cols_final = 4
        n_rows_final = 10

        # Calculate actual grid center for title positioning
        grid_width = (n_cols_final - 1) * grid_spacing
        grid_height = (n_rows_final - 1) * grid_spacing
        grid_visual_center_y = grid_top_y - grid_height / 2

        # Title goes centered above the grid, closer to the icons
        title_target_y = grid_top_y + 0.35

        move_anims = [
            self.title_main.animate.scale(0.7).move_to([grid_center_x, title_target_y, 0]),
            self.title_sub.animate.scale(0.7).move_to([grid_center_x, title_target_y - 0.3, 0]),
        ]

        # Arrange channels in a tighter column on the left (shrink + slide)
        for i, circle in enumerate(self.channels_group):
            row = i % n_rows_final
            col = i // n_rows_final
            new_x = grid_center_x + (col - (n_cols_final - 1) / 2) * grid_spacing
            new_y = grid_top_y - row * grid_spacing
            move_anims.append(circle.animate.scale(0.4).move_to([new_x, new_y, 0]))

        for i, img in enumerate(self.channel_images):
            row = i % n_rows_final
            col = i // n_rows_final
            new_x = grid_center_x + (col - (n_cols_final - 1) / 2) * grid_spacing
            new_y = grid_top_y - row * grid_spacing
            move_anims.append(img.animate.scale(0.4).move_to([new_x, new_y, 0]))

        self.play(LaggedStart(*move_anims, lag_ratio=0.02), run_time=1.5)
        self.wait(0.3)

    def part2_episodes(self):
        """Part 2: Episodes - thumbnails fly from channels to year timeline on right."""
        import os

        new_title = Text("48,724 Episodes (2020-2025)", font_size=28, color=WHITE)
        new_title.to_edge(UP, buff=0.4)

        self.play(FadeIn(new_title), run_time=0.6)

        # Year axis on the FAR RIGHT side (2020 at top, 2025 at bottom)
        year_line_x = 6.5
        year_line = Line([year_line_x, 2.4, 0], [year_line_x, -2.5, 0], color=GRAY, stroke_width=1.5)
        year_line_ticks = VGroup()
        year_line_labels = VGroup()
        years = ["2020", "2021", "2022", "2023", "2024", "2025"]
        for i, year in enumerate(years):
            y = 2.4 - i * (4.9 / 5)
            tick = Line([year_line_x - 0.1, y, 0], [year_line_x, y, 0], color=GRAY, stroke_width=1)
            lbl = Text(year, font_size=12, color=GRAY)
            lbl.next_to(tick, LEFT, buff=0.1)
            year_line_ticks.add(tick)
            year_line_labels.add(lbl)
        self.play(Create(year_line), FadeIn(year_line_ticks), FadeIn(year_line_labels), run_time=0.6)

        # Load video thumbnails
        script_dir = os.path.dirname(os.path.abspath(__file__))
        thumb_dir = os.path.join(script_dir, "video_thumbnails")
        thumb_files = sorted([
            f for f in os.listdir(thumb_dir)
            if f.lower().endswith((".jpg", ".png"))
        ])[:100]  # Use up to 100 thumbnails

        # Episode distribution by year
        year_episode_counts = [3000, 5000, 8000, 10000, 12000, 10724]  # = 48,724
        year_rect_counts = [c // 100 for c in year_episode_counts]  # 30, 50, 80, 100, 120, 107

        # Create all episode rectangles positioned by year (filling LEFT of the year line)
        all_episodes = VGroup()
        rect_height = 0.12
        rect_width = 0.18
        year_rects = [[] for _ in years]

        for year_idx, n_rects in enumerate(year_rect_counts):
            year_y = 2.4 - year_idx * (4.9 / 5)
            x_right_edge = year_line_x - 0.5  # Start from near the year line
            cols_rect = 28
            spacing_rect_x = 0.18
            spacing_rect_y = 0.15
            for j in range(n_rects):
                rect = Rectangle(
                    width=rect_width,
                    height=rect_height,
                    color=EPISODE_BLUE,
                    fill_opacity=0.7,
                    stroke_width=0
                )
                col_in_year = j % cols_rect
                row_in_year = j // cols_rect
                # Fill from right to left (near year line going toward center)
                x = x_right_edge - col_in_year * spacing_rect_x
                y = year_y - 0.15 - row_in_year * spacing_rect_y
                rect.move_to([x, y, 0])
                all_episodes.add(rect)
                year_rects[year_idx].append(rect)

        # Assign thumbnails to rectangles (distribute across years)
        thumb_assignments = []  # (thumb_file, target_rect, year_idx)
        temp_year_rects = [list(lst) for lst in year_rects]
        for i, thumb_file in enumerate(thumb_files):
            year_idx = i % len(years)
            if temp_year_rects[year_idx]:
                rect = temp_year_rects[year_idx].pop(0)
                thumb_assignments.append((thumb_file, rect, year_idx))

        # Create thumbnails starting near the channel icons (left side)
        # Channel grid is centered at x=-5.5, so spawn just to the right of it
        spawn_base_x = -3.5
        demo_thumbs = Group()
        for i, (thumb_file, target_rect, year_idx) in enumerate(thumb_assignments):
            thumb_path = os.path.join(thumb_dir, thumb_file)
            img = ImageMobject(thumb_path)
            img.height = 0.5
            img.width = 0.7
            # Spawn in a cluster near channels, distributed by year
            spawn_y = 2.0 - year_idx * 0.7  # Spread vertically by year
            spawn_x_offset = (i % 10) * 0.15  # Small horizontal spread
            img.move_to([spawn_base_x + spawn_x_offset, spawn_y, 0])
            demo_thumbs.add(img)

        # First fade in thumbnails near the channels
        self.play(
            LaggedStart(*[FadeIn(img, scale=0.5) for img in demo_thumbs], lag_ratio=0.02),
            run_time=1.5
        )

        # Then fly them to their target positions with acceleration
        fly_anims = []
        for img, (thumb_file, target_rect, year_idx) in zip(demo_thumbs, thumb_assignments):
            fly_anims.append(
                img.animate(rate_func=rate_functions.ease_in_out_quad).move_to(target_rect.get_center()).scale(0.35)
            )

        self.play(
            LaggedStart(*fly_anims, lag_ratio=0.02),
            run_time=2.5
        )

        # Transform thumbnails into blue rectangles and fill in the rest
        transform_anims = []
        used_rects = set()
        for img, (_, target_rect, _) in zip(demo_thumbs, thumb_assignments):
            transform_anims.append(AnimationGroup(FadeOut(img), FadeIn(target_rect)))
            used_rects.add(id(target_rect))

        # Remaining rectangles that weren't assigned thumbnails
        remaining_rects = [r for r in all_episodes if id(r) not in used_rects]

        self.play(
            LaggedStart(
                *transform_anims,
                *[FadeIn(r, scale=0.5) for r in remaining_rects],
                lag_ratio=0.002,
            ),
            run_time=2,
        )

        self.wait(0.6)

        # Store for next transition
        self.episodes_group = all_episodes
        self.timeline_group = VGroup(year_line, year_line_labels)
        self.current_title = new_title

    def part3_segmentation(self):
        """Part 3: Zoom into one episode, break it into segments."""
        new_title = Text("1.1M Segments (~70 sec each)", font_size=28, color=WHITE)
        new_title.to_edge(UP, buff=0.4)

        # First: Move everything from part2 off screen to the left
        self.play(
            self.channels_group.animate.shift(LEFT * 10).set_opacity(0),
            self.channel_images.animate.shift(LEFT * 10).set_opacity(0),
            self.title_main.animate.shift(LEFT * 10).set_opacity(0),
            self.title_sub.animate.shift(LEFT * 10).set_opacity(0),
            self.timeline_group.animate.shift(LEFT * 10).set_opacity(0),
            self.episodes_group.animate.shift(LEFT * 10).set_opacity(0),
            ReplacementTransform(self.current_title, new_title),
            run_time=1.2,
        )

        self.wait(0.3)

        # Now bring in a single episode from the left to zoom into
        big_episode = Rectangle(
            width=1.0,
            height=0.6,
            color=EPISODE_BLUE,
            fill_opacity=0.5,
            stroke_width=2,
        )
        big_episode.move_to([-8, 0, 0])  # Start off-screen left
        self.add(big_episode)

        # Fly in and grow the episode
        self.play(
            big_episode.animate.scale_to_fit_height(5.0).scale_to_fit_width(6.0).move_to([0, 0, 0]),
            run_time=1.2,
        )
        big_episode.set_fill(opacity=0.18).set_stroke(EPISODE_BLUE, width=2)

        # Transcript inside the enlarged episode
        # Simulate an 80-minute transcript block with dense text
        long_text = (
            "Segment 15472253 — lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit "
            "in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
            "mollit anim id est laborum. " * 8  # repeat to convey volume
        )
        transcript = Paragraph(
            long_text,
            font_size=16,
            color=WHITE,
            alignment="left",
            line_spacing=0.7,
        )
        transcript.width = 5.5
        transcript.move_to(big_episode.get_center())

        self.play(FadeIn(transcript), run_time=0.6)
        self.wait(1.0)

        # Highlight a single segment from the long transcript
        segment_text = Paragraph(
            "A single segment is extracted from this long transcript (~70 seconds).",
            font_size=14,
            color=WHITE,
            alignment="left",
            line_spacing=0.7,
        )
        segment_text.width = 4.5
        segment_text.next_to(big_episode, RIGHT, buff=0.4).align_to(big_episode, UP)

        seg_square = Square(
            side_length=0.6,
            color=EPISODE_BLUE,
            fill_opacity=0.35,
            stroke_width=1.5,
        )
        seg_square.next_to(big_episode, DOWN, buff=0.4).align_to(big_episode, LEFT)

        self.play(FadeIn(segment_text), FadeIn(seg_square, scale=0.3), run_time=0.8)

        explain = Text("Each segment becomes a data point", font_size=16, color=WHITE)
        explain.move_to([0, 3.2, 0])

        self.play(
            FadeOut(transcript),
            big_episode.animate.set_fill(opacity=0.1),
            run_time=0.6
        )
        self.play(FadeIn(explain), run_time=0.4)

        # Transition directly into full corpus: move cluster to top-left and fill in
        corpus_explain = Text("Full corpus: 1.1M segments", font_size=16, color=WHITE)
        corpus_explain.move_to([0, 3.2, 0])
        self.play(ReplacementTransform(explain, corpus_explain), run_time=0.4)

        n_cols, n_rows = 55, 28
        spacing_x = 0.12
        spacing_y = 0.12
        origin = np.array([-6.2, 3.0, 0])
        positions = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = origin[0] + col * spacing_x
                y = origin[1] - row * spacing_y
                positions.append([x, y, 0])

        n_dots = len(positions)
        n_candidates = int(n_dots * (self.stage1_candidates / self.total_segments))

        np.random.seed(42)
        indices = list(range(n_dots))
        np.random.shuffle(indices)
        candidate_indices = set(indices[:n_candidates])

        all_dots = VGroup()
        self.candidate_dots = VGroup()
        self.non_candidate_dots = VGroup()

        corpus_square_size = 0.1
        shrink_factor = corpus_square_size / seg_square.side_length
        seg_target = positions[0]
        self.play(seg_square.animate.move_to(seg_target).scale(shrink_factor), run_time=0.6)
        all_dots.add(seg_square)
        self.candidate_dots = VGroup(seg_square)
        self.non_candidate_dots = VGroup()

        new_square_anims = []
        for idx in range(1, n_dots):
            sq = Square(
                side_length=corpus_square_size,
                color=SEGMENT_GRAY,
                fill_opacity=0.5,
                stroke_width=1,
            )
            sq.move_to(positions[idx])
            all_dots.add(sq)
            if idx in candidate_indices:
                self.candidate_dots.add(sq)
            else:
                self.non_candidate_dots.add(sq)
            new_square_anims.append(FadeIn(sq, scale=0.3))

        self.play(
            LaggedStart(*new_square_anims, lag_ratio=0.0015),
            run_time=2
        )

        self.wait(0.5)

        # Store for next phase
        self.all_dots = all_dots
        self.corpus_explain = corpus_explain
        self.current_title = new_title

    def part4_classification_gates(self):
        """Part 4: Flow through classification gates with dashed barriers."""

        # Precompute dashed barriers and final block positions
        barrier_x = [-2.6, 0.2, 3.0]
        barriers = VGroup(*[
            DashedLine([x, -3, 0], [x, 3, 0], color=BARRIER_GLOW, stroke_width=2, dash_length=0.2)
            for x in barrier_x
        ])

        # ===== STAGE 1: Does it sound like misogyny? =====
        stage1_title = Text("Stage 1: Does it sound like misogyny?", font_size=24, color=CANDIDATE_GREEN)
        stage1_title.to_edge(UP, buff=0.3)
        stage1_sub = Text("Initial semantic filter (231,905 flagged)", font_size=14, color=GRAY)
        stage1_sub.next_to(stage1_title, DOWN, buff=0.1)

        self.play(
            ReplacementTransform(self.current_title, stage1_title),
            ReplacementTransform(self.corpus_explain, stage1_sub),
            FadeIn(barriers[0]),
            run_time=0.6
        )

        block1_positions = self.layout_block(self.all_dots, cols=20, center=np.array([-4.5, 0.2, 0]), spacing=0.14)
        self.play(
            *[
                d.animate.move_to(block1_positions[i]).set_fill(
                    CANDIDATE_GREEN if d in self.candidate_dots else SEGMENT_GRAY,
                    opacity=0.9 if d in self.candidate_dots else 0.1,
                ).set_stroke(width=1, opacity=0.35)
                for i, d in enumerate(self.all_dots)
            ],
            run_time=1.3,
        )

        self.wait(0.3)

        # ===== STAGE 2: Is the discussion primarily about gender? =====
        stage2_title = Text("Stage 2: Is the discussion primarily about gender?", font_size=24, color=WHITE)
        stage2_title.to_edge(UP, buff=0.3)
        stage2_sub = Text("47,393 remain", font_size=14, color=GRAY)
        stage2_sub.next_to(stage2_title, DOWN, buff=0.1)

        dots_list = list(self.candidate_dots)
        np.random.shuffle(dots_list)
        n_excl = int(len(dots_list) * self.stage3_excluded / self.stage1_candidates)
        self.dots_excluded_stage2 = VGroup(*dots_list[:n_excl])
        self.dots_passed_stage2 = VGroup(*dots_list[n_excl:])

        self.play(
            ReplacementTransform(stage1_title, stage2_title),
            ReplacementTransform(stage1_sub, stage2_sub),
            FadeIn(barriers[1]),
            run_time=0.6
        )

        excl_positions = self.layout_block(self.dots_excluded_stage2, cols=12, center=np.array([-1.0, 1.2, 0]), spacing=0.15)
        pass_positions = self.layout_block(self.dots_passed_stage2, cols=20, center=np.array([-1.0, -0.4, 0]), spacing=0.14)

        self.play(
            *[d.animate.move_to(pos).set_fill(EXCLUDED_RED, opacity=0.9) for d, pos in zip(self.dots_excluded_stage2, excl_positions)],
            *[d.animate.move_to(pos).set_fill(CANDIDATE_GREEN, opacity=0.9) for d, pos in zip(self.dots_passed_stage2, pass_positions)],
            run_time=1.3,
        )

        excl_count = Text("184,512 excluded", font_size=11, color=EXCLUDED_RED)
        excl_count.move_to([-1.0, 2.7, 0])
        self.play(FadeIn(excl_count), run_time=0.3)

        self.wait(0.3)

        # ===== STAGE 3: Do the speakers hold the position? =====
        stage3_title = Text("Stage 3: Do the speakers hold the position?", font_size=24, color=WHITE)
        stage3_title.to_edge(UP, buff=0.3)
        stage3_sub = Text("41,539 hold views", font_size=14, color=GRAY)
        stage3_sub.next_to(stage3_title, DOWN, buff=0.1)

        passed_list = list(self.dots_passed_stage2)
        np.random.shuffle(passed_list)
        n_reject = int(len(passed_list) * self.stage4_rejects / self.stage3_passed)
        self.dots_holds = VGroup(*passed_list[n_reject:])
        self.dots_rejects = VGroup(*passed_list[:n_reject])

        self.play(
            ReplacementTransform(stage2_title, stage3_title),
            ReplacementTransform(stage2_sub, stage3_sub),
            FadeIn(barriers[2]),
            FadeOut(excl_count),
            run_time=0.6
        )

        reject_positions = self.layout_block(self.dots_rejects, cols=10, center=np.array([2.2, 1.0, 0]), spacing=0.15)
        hold_positions = self.layout_block(self.dots_holds, cols=20, center=np.array([2.4, -0.4, 0]), spacing=0.14)

        self.play(
            *[d.animate.move_to(pos).set_fill(ORANGE, opacity=0.9) for d, pos in zip(self.dots_rejects, reject_positions)],
            *[d.animate.move_to(pos).set_fill(INCLUDED_GREEN, opacity=0.9) for d, pos in zip(self.dots_holds, hold_positions)],
            run_time=1.3,
        )

        reject_count = Text("5,854 uncertain", font_size=11, color=ORANGE)
        reject_count.move_to([2.2, 2.6, 0])
        self.play(FadeIn(reject_count), run_time=0.3)

        self.wait(0.4)

        # Final state for next part
        self.barriers = barriers
        self.rejection_labels = VGroup(reject_count)
        self.current_title = stage3_title
        self.current_sub = stage3_sub

    def part5_final_result(self):
        """Part 5: Consolidate final included segments."""
        final_title = Text("Final Validated Corpus", font_size=28, color=INCLUDED_GREEN)
        final_title.to_edge(UP, buff=0.3)
        final_count = Text("41,539 segments (3.6% of corpus)", font_size=18, color=WHITE)
        final_count.next_to(final_title, DOWN, buff=0.15)

        # Combine all included dots
        all_included = VGroup(*self.dots_holds)

        # Create final collection box
        final_box = Rectangle(
            width=3, height=2,
            color=INCLUDED_GREEN,
            fill_opacity=0.1,
            stroke_width=3
        )
        final_box.move_to([0, -0.5, 0])

        self.play(
            ReplacementTransform(self.current_title, final_title),
            ReplacementTransform(self.current_sub, final_count),
            FadeOut(self.barriers),
            FadeOut(self.rejection_labels),
            FadeOut(self.dots_excluded_stage2),
            FadeOut(self.dots_rejects),
            FadeOut(self.non_candidate_dots),
            Create(final_box),
            run_time=1
        )

        # Move all included dots into the box
        np.random.seed(99)
        self.play(
            *[d.animate.move_to(
                final_box.get_center() + np.array([
                    np.random.uniform(-1.2, 1.2),
                    np.random.uniform(-0.8, 0.8),
                    0
                ])
            ) for d in all_included],
            run_time=1.5
        )

        # Summary stats
        summary = Text(
            "41,539 segments retained where speakers hold the position",
            font_size=14,
            color=GRAY
        )
        summary.move_to([0, -2.5, 0])

        self.play(FadeIn(summary), run_time=0.5)

        self.wait(3)

    def create_barrier(self, x_pos):
        """Create an abstract glowing barrier (force-field style)."""
        # Main line
        line = Line(
            [x_pos, -3, 0],
            [x_pos, 3, 0],
            color=BARRIER_GLOW,
            stroke_width=2
        )

        # Glow effect (multiple fading lines)
        glow = VGroup()
        for i in range(1, 4):
            glow_line = Line(
                [x_pos, -3, 0],
                [x_pos, 3, 0],
                color=BARRIER_GLOW,
                stroke_width=2 + i * 2,
                stroke_opacity=0.3 / i
            )
            glow.add(glow_line)

        return VGroup(glow, line)
