# Analysis Step 3: Create a Report on Cited Articles
This report drills down to the next level of detail. Instead of just showing the website, it focuses on the specific article URLs that were cited. This helps you answer: "Which exact pages are being used as sources?"

### Using the Filters
Before running the report, you can use the filter controls (like dropdown menus and search boxes) to narrow down your dataset. You can look for a specific article, domain, or prompt to see how individual URLs are performing.

### Report Metrics Guide
Here’s a guide to what each column in the report means:

- **Article URL**: The specific, full URL of the article that was cited.
- **Article Title**: The title of the cited article.
- **Domain**: The main website address (e.g., `theverge.com`) the article belongs to.
- **Appearances**: The total number of times this specific article was cited in the filtered results.
- **Prompts w/ Citations**: How many of your unique prompts led to this article being cited.
- **Prompt Reach %**: The percentage of *all* unique prompts in your filtered view that cited this article.
- **Outputs w/ Citations**: The number of individual AI responses that included a citation for this article.
- **Output Hit Rate %**: The percentage of *all* AI responses that cited this article.
- **Avg / Median / Std Position**: Statistics on the article's rank in the list of sources (lower is better). The standard deviation (`std`) shows if the ranking is consistent.
- **Top 3 Rate %**: The percentage of time this article appeared in the top 3 citation spots.
- **First / Last Seen**: The timestamps for the first and most recent time the article appeared in your data.
