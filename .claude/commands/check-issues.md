# Check Core Repository GitHub Issues

Review all open GitHub issues in the core repository and provide a prioritized recommendation.

## Instructions

1. **Fetch all open issues**:
   ```bash
   cd ~/signal4/core && gh issue list --state open --limit 30
   ```

2. **Get details** on the top 5 most recent/relevant issues:
   ```bash
   cd ~/signal4/core && gh issue view <number> --json title,body,labels,createdAt
   ```

3. **Categorize issues** by type:
   - **Pipeline bugs** (processing failures, stuck content)
   - **Data quality** (embeddings, migrations, integrity)
   - **API/Backend** (endpoints, performance)
   - **Infrastructure** (workers, orchestration)
   - **Technical debt** (cleanup, refactoring)

4. **Analyze dependencies**:
   - Which issues block other issues?
   - Which issues are partially complete?
   - What has impact on production data quality?

5. **Output format**:
   - Summary table of all open issues
   - Top recommendation with detailed reasoning
   - 2-3 alternatives based on different priorities (quick win, high impact, etc.)

## Core-Specific Priorities

High priority for core:
- Data integrity issues (bad embeddings, stuck content)
- Pipeline stability (worker failures, processing errors)
- Schema migrations in progress
- Deprecation completions (speaker turns -> sentences)

Medium priority:
- Warning/logging cleanup
- Code cleanup (batch mode removal)
- New pipeline features

Lower priority:
- Testing tasks
- Documentation
