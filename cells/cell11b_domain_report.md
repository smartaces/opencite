# Analysis Step 2: Create a Report on Cited Websites
This report gives you a high-level view of your results by grouping citations by the main website (e.g., `theverge.com` or `wikipedia.org`). It's designed to help you answer the question: "Which websites are appearing most often as authorities for my topics?"

### Using the Filters
Before running the report, you can use the filter controls (like dropdown menus and search boxes) to narrow down your dataset. This allows you to focus on a specific scenario, persona, prompt, or time frame. The metrics in the report will update based on your filtered selection.

### Report Metrics Guide
Here’s a guide to what each column in the report means:

- **Domain**: The main website address (e.g., `theverge.com`) that was cited.
- **Appearances**: The total number of times this domain was cited in the filtered results.
- **Unique Pages**: The number of different, specific articles from this domain that were cited.
- **Prompts w/ Citations**: How many of your unique prompts led to this domain being cited.
- **Prompt Reach %**: The percentage of *all* unique prompts in your filtered view that cited this domain. This shows how widely the domain is seen as an authority across different topics.
- **Outputs w/ Citations**: The number of individual AI responses that included a citation from this domain.
- **Output Hit Rate %**: The percentage of *all* AI responses that cited this domain. This shows how frequently it appears, especially when you re-run the same prompt.
- **Avg Cites/Output**: The average number of citations a domain gets in a single AI response where it appears. A high number means it's often cited multiple times.
- **Avg / Median / Std Position**: Statistics on the citation's rank in the list of sources (lower is better). The standard deviation (`std`) shows if the ranking is consistent or volatile.
- **Top 3 Rate %**: The percentage of time this domain appeared in the top 3 citation spots.
- **Consistency**: A score based on the ranking's standard deviation. A higher score means the domain's rank is more stable and predictable.
- **First / Last Seen**: The timestamps for the first and most recent time the domain appeared in your data, which is useful for seeing if a source is still current.
