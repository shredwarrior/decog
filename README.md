# Argument Analyzer — Cognition Today

A web application that analyzes arguments, opinions, and conclusions for logical fallacies, cognitive biases, cognitive distortions, and philosophical razors. It uses a metadata-aware LLM pipeline and deterministic scoring to produce a structured analysis report.

## Features

- **Metadata-aware analysis pipeline**: Metadata extraction (Call 1) feeds enriched context into the main analysis (Call 2) for higher accuracy
- **Short-deduction assist (conditional LLM call)**: For short inputs, an extra formal-logic hint checks concise math/philosophy-style deductions
- **23-feature metadata extraction**: Each claim is tagged with structural and epistemic quality features (face validity, speculation level, evidence sufficiency, causal chain length, exemplar type, symmetry/proportionality patterns, etc.)
- **Logical Fallacy Detection**: 17 fallacies with weighted penalties
- **Cognitive Bias Analysis**: 21 biases with weighted penalties
- **Cognitive Distortion Detection**: 12 distortions with weighted penalties
- **Philosophical Razor Evaluation**: 6 razors (Occam's, Hanlon's, Hitchens's, Sagan Standard, Newton's Flaming Laser Sword, Popper's Falsifiability)
- **Four-score dashboard**: Overall Strength + Bias Score + Testability + Logic Integrity
- **Logic-first composite scoring**: Deterministic logic variables + science-reasoning proxy + connector signals, with mode-aware lanes
- **Mode-aware weighting**: Empirical vs deductive/speculative vs fictional/world-internal weighting and guardrails
- **UI-safe logic interpretation**: UI uses an effective logic score, while raw logic integrity remains available for diagnostics
- **PDF download**: Branded analysis report
- **Social sharing**: Image snapshot of results via html2canvas
- **Email subscription**: Unlocks 2,000-word input (free tier: 300 words)
- **SQLite storage**: All analyses stored for future use

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   SESSION_SECRET_KEY=your_random_secret_here
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Application**
   Open your browser at `http://localhost:5000`

## Project Structure

```
argument-analyser/
├── app.py                 # Flask app, routes, session handling
├── analyzer.py            # Two-stage LLM pipeline + scoring
├── engagement.py          # SQLite DB, PDF, share, subscribe
├── definitions.json       # Fallacies, biases, distortions, razors
├── SYSTEM_LOGIC.txt       # Full documentation of logic and taxonomy
├── requirements.txt       # Python dependencies
├── Procfile               # Production server config (gunicorn)
├── .env                   # Environment variables (not committed)
├── .gitignore
├── README.md
├── templates/
│   ├── index.html         # Main frontend
│   └── share.html         # Minimal share page with OG meta tags
└── static/
    └── style.css
```

## How It Works

1. User enters an argument (max 300 words, or 2,000 if subscribed)
2. **Call 1** (metadata): LLM extracts structural + epistemic features per claim
3. Metadata is aggregated into a context summary with flags
4. **Call 2** (analysis): LLM detects issues and evaluates razors using metadata context
5. **Conditional short-deduction call** (short inputs only): LLM checks concise formal deduction quality
6. Results are normalized, filtered by confidence thresholds, and scored:
   - Legacy structural score from weighted issue penalties + capped razor rewards
   - Logic variable composite score from deterministic structural signals
   - Science proxy / connector hints raise logic confidence for concise rigorous reasoning
   - Mode detection (`empirical`, `deductive`, `speculative`, `fictional`) adjusts penalty and blend weights
   - Contradiction/overclaim guardrails cap inflated rhetorical scores
   - UI shows effective logic score; raw logic score remains in diagnostics
   - Final score clamped 0–100
7. Analysis is auto-saved to SQLite with a UUID
8. User can download PDF, share image, or copy share link

## API Endpoints

- `GET /` — Main page
- `POST /api/analyze` — Analyze an argument (`{"argument": "..."}`)
- `GET /api/definitions` — All definitions from JSON
- `POST /api/subscribe` — Email subscription (`{"email": "..."}`)
- `POST /api/analysis-feedback` — Prototype calibration feedback (`analysis_id`, score/suggestion labels, optional revised argument)
- `GET /api/download-pdf/<id>` — PDF download for a stored analysis
- `GET /api/share/<id>` — Public share page with OG tags

## Deployment (Railway)

1. Push to GitHub
2. Connect repo to Railway
3. Set environment variables in Railway dashboard:
   - `OPENAI_API_KEY`
   - `SESSION_SECRET_KEY`
4. Railway auto-detects Python + Procfile and deploys

## Notes

- Requires an OpenAI API key (GPT-4o-mini recommended)
- Scoring and definitions are customizable via `definitions.json` and environment variables
- Calibration/debug logs are emitted as JSONL in `logs/` (`analysis_artifacts.jsonl`, `analysis_feedback.jsonl`) unless overridden by env vars
- See `SYSTEM_LOGIC.txt` for full documentation of the taxonomy and scoring framework
- Run `python calibration/test_fiction_logic.py` to execute the fiction logic discrimination pilot and verify gate metrics
