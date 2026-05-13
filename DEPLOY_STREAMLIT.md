# Deploying the dashboard to Streamlit Community Cloud

The Streamlit app is already complete at `app.py`. To publish it for free
on `*.streamlit.app`, follow the steps below.

## Prerequisites

- A Streamlit Community Cloud account at <https://share.streamlit.io/>
  (free, sign in with GitHub).
- The repository must be **public on GitHub** (already the case).

## Deploy

1. Go to <https://share.streamlit.io/new>.
2. Pick the `rohanfosse/bikeshare-data-explorer` repo.
3. Set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `gbfs-audit` (final URL : <https://gbfs-audit.streamlit.app>)
4. Click **Deploy**.

That's it. The app builds in 3–5 minutes from `requirements.txt` and is
live at the URL above.

## Cite in the paper

Once deployed, add to the Data and code availability section :

> An interactive dashboard exposing every catalogue page, IMD ranking,
> and Use-Case query is hosted free at
> <https://gbfs-audit.streamlit.app>.

## Local preview

```bash
streamlit run app.py
```

opens at <http://localhost:8501>.

## Memory / build notes

- Streamlit Cloud free tier : 1 GB RAM. The 6.6 MB parquet loads
  comfortably within that limit.
- The first cold start takes ~30 s as the catalogue is loaded into
  the `@st.cache_data` layer.
- Subsequent loads are instantaneous (cached for 1 h via `ttl=3600`
  in `utils/data_loader.py`).

## Public URL stability

The URL `gbfs-audit.streamlit.app` is reserved at deployment time and
stays available as long as the repo + the app are public and the app
sees at least one weekly visit (Streamlit Cloud hibernates idle apps
after seven days but resumes them on demand).
