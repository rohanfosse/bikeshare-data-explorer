# Publishing the project page via GitHub Pages

The `docs/` directory is a Jekyll-ready project page. Enable it once
on the GitHub repository settings and every push to `main` rebuilds
the public site automatically.

## One-time setup

1. Open <https://github.com/rohanfosse/bikeshare-data-explorer/settings/pages>.
2. Under **Build and deployment** → **Source**, select
   **Deploy from a branch**.
3. Set **Branch** to `main` and **Folder** to `/docs`.
4. Click **Save**.

The site builds within ~60 s and lands at:

<https://rohanfosse.github.io/bikeshare-data-explorer/>

That URL is the one cited in `docs/index.md` and matches the
`url` field of `_config.yml`.

## Local preview (optional)

To preview the rendered site before pushing :

```bash
gem install bundler
cd docs
bundle init && bundle add jekyll
bundle exec jekyll serve --baseurl ""
```

Opens at <http://localhost:4000>.

## Updating

Every push to `main` that touches anything under `docs/` triggers
a rebuild. No manual deploy step is required.
