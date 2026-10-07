# 📄 ATS Resume Analyzer

A Streamlit app that estimates how well a resume will perform in an Applicant Tracking System (ATS) and suggests specific improvements, powered by Google's Gemini Flash model.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- Optionally paste a **job description** for a targeted keyword match
- **ATS score (0-100)** with a weighted breakdown:
  keywords & relevance (30%), formatting & parsability (20%), section completeness (15%), impact & achievements (20%), language & readability (15%)
- Strengths, **missing keywords**, **prioritised improvements** and **rewritten bullet points**
- Download the full report as Markdown

> The score is an AI estimate. No universal ATS score exists, and real systems differ.

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Install Python 3.10+ and clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
4. Provide the key in **one** of these ways:
   - Paste it into the app's sidebar when it runs, or
   - Set an environment variable:
     - macOS/Linux: `export GEMINI_API_KEY="your-key"`
     - Windows (PowerShell): `$env:GEMINI_API_KEY="your-key"`
   - Or create `.streamlit/secrets.toml` (never commit this file):
     ```toml
     GEMINI_API_KEY = "your-key"
     ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (a public repo works with the free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → choose your repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Choosing the model

The sidebar lets you pick a model (default `gemini-flash-latest`, an alias for the newest Flash model). To set another default, add this to your secrets:

```toml
GEMINI_MODEL = "gemini-3.6-flash"
```

Model names change over time. See the [Gemini models page](https://ai.google.dev/gemini-api/docs/models) for current names.

## Privacy and security

- Resume text is sent to the Gemini API for analysis. Review Google's API terms, especially for the free tier, before uploading sensitive documents.
- Never commit your API key. Keep it in Streamlit **Secrets** or an environment variable.
- Scanned or image-only PDFs have no extractable text. The app will warn you (and ATS systems cannot read them either).

## Troubleshooting

| Problem | Fix |
|---|---|
| "API key was rejected" | Check the key at aistudio.google.com |
| "Model name was not found" | Pick a different model in the sidebar |
| "Rate limit or quota reached" | Wait a minute, or use another model |
| "Very little text could be extracted" | Export a text-based PDF or DOCX, not a scan |
