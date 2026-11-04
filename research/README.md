# Engagement research

Two tools for checking the pipeline's defaults against real top-performing clips in **your** niche.

## YouTube Shorts: `benchmark.py`

Run this on your own machine; YouTube blocks most cloud servers. It reads **metadata and auto-captions only** and never downloads videos.

```bash
python research/benchmark.py \
  https://www.youtube.com/@ChannelA/shorts \
  https://www.youtube.com/@ChannelB/shorts \
  --max-per-source 60 --rank-by views_per_day
```

It writes three files to `research/out/`:
- **`shorts.csv`**: one row per Short, with views, likes, comments, length, upload date, engagement rate, views per day, and the opening line. It also flags whether the opening line is a question, has a number, says "you", or is contrarian.
- **`summary.md`**: compares the top 10% with the rest on median length, time to the first spoken word, and how often each type of hook appears.
- **`top.jsonl`**: the openings of the top 10%. Feed them to clip selection as inspiration:

  ```bash
  python main.py episode.mp4 --mode local --examples research/out/top.jsonl
  ```

Tips:
- Pick 3–6 channels that post clips of content like yours (podcast clips from your genre), and get at least 30 Shorts in total.
- Use `--rank-by views_per_day` so old videos don't dominate.
- If YouTube asks you to sign in, add `--cookies-from-browser chrome` (or firefox).
- Keep request volumes modest and follow YouTube's Terms of Service.

## Instagram Reels: `instagram_reels_template.csv`

Instagram doesn't allow scraping, so this one is manual. Log 20–30 top Reels in your niche, plus your own Reels Insights (sends and saves are only visible for your own posts). Columns cover the things that matter for Reels: length, the first spoken line, on-screen text in the first 3 s, caption style, layout, and whether it loops.

Instagram weighs **sends** (DM shares) heavily, so note which Reels you'd forward to a friend and why.

## Reading the results

The output shows correlation, not causation. Use it to tune `--target-length`, the hook rules in `shorts_generator/highlights.py` (`PLATFORM_SIGNALS`), and the `--examples` file. Re-run it every month or two, because trends drift.
